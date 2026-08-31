"""Collect a NexArm demonstration from Cartesian waypoints in an unchanged LIBERO task."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "LIBERO")]

import libero_nexarm  # noqa: F401, E402 - registers the custom robot and gripper
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


CONTROL_HZ = 100
RECORD_HZ = 20
SIM_STEPS = 5
RECORD_EVERY = CONTROL_HZ // RECORD_HZ


def make_env(bddl_path):
    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        robots=["NexArm"],
        controller="JOINT_POSITION",
        initialization_noise=None,
        control_freq=RECORD_HZ,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=128,
        camera_widths=128,
    )


def settle(env):
    for _ in range(RECORD_HZ):
        env.step(np.r_[np.zeros(5), 1.0])


def resolve_waypoints(spec, env):
    waypoints = []
    for item in spec["waypoints"]:
        if "xyz" in item:
            xyz = np.asarray(item["xyz"], dtype=float)
        else:
            body_id = env.env.obj_body_id[item["body"]]
            xyz = env.sim.data.body_xpos[body_id] + np.asarray(item["offset"])
        waypoints.append((xyz, float(item["gripper"]), float(item["duration"])))
    return waypoints


def solve_ik(env, targets):
    robot = env.robots[0]
    model = env.sim.model._model
    clone = mujoco.MjData(model)
    clone.qpos[:] = env.sim.data.qpos
    qpos = np.asarray(robot._ref_joint_pos_indexes)
    joint_ids = np.asarray(robot._ref_joint_indexes)
    low, high = model.jnt_range[joint_ids].T
    site = robot.eef_site_id
    q = robot._joint_positions.copy()
    result = []

    def residual(value, target, seed):
        clone.qpos[qpos] = value
        mujoco.mj_kinematics(model, clone)
        mujoco.mj_comPos(model, clone)
        approach = clone.site_xmat[site].reshape(3, 3)[:, 1]
        return np.r_[
            100.0 * (clone.site_xpos[site] - target),
            2.0 * (approach - np.array([0.0, 0.0, 1.0])),
            0.05 * (value - seed),
        ]

    for target in targets:
        q = least_squares(
            residual, q, args=(target, q.copy()), bounds=(low, high), max_nfev=300
        ).x
        error = np.linalg.norm(residual(q, target, q)[:3]) / 100.0
        if error > 0.01:
            raise RuntimeError(f"IK misses waypoint {target.round(3)} by {error:.3f} m")
        result.append(q.copy())
    return result


def observe(env, gripper):
    env._post_process()
    env._update_observables(force=True)
    obs = env.env._get_observations()
    robot = env.robots[0]
    joint_pos = robot._joint_positions.copy()
    joint_vel = robot._joint_velocities.copy()
    gripper_pos = env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes].copy()
    ee_pos = env.sim.data.site_xpos[robot.eef_site_id].copy()
    return {
        "front_rgb": obs["agentview_image"].copy(),
        "wrist_rgb": obs["robot0_eye_in_hand_image"].copy(),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "gripper_pos": gripper_pos,
        "ee_pos": ee_pos,
        "proprio": np.r_[joint_pos, joint_vel, gripper_pos, ee_pos],
        "gripper": gripper,
    }


def execute(env, waypoints, joint_targets):
    robot = env.robots[0]
    robot.controller.kp[:] = 150.0
    robot.controller.kd[:] = 2.0 * np.sqrt(robot.controller.kp)
    frames = [observe(env, 1.0)]
    gripper_start = 1.0

    for (_, gripper_goal, duration), q_goal in zip(waypoints, joint_targets):
        q_start = robot._joint_positions.copy()
        steps = max(1, round(duration * CONTROL_HZ))
        for step in range(1, steps + 1):
            t = step / steps
            blend = t * t * (3.0 - 2.0 * t)
            q_target = q_start + blend * (q_goal - q_start)
            gripper = gripper_start + blend * (gripper_goal - gripper_start)
            robot.controller.set_goal(np.zeros(5), set_qpos=q_target)
            for _ in range(SIM_STEPS):
                robot.control(np.r_[np.zeros(5), gripper], policy_step=False)
                env.sim.step()
            if step % RECORD_EVERY == 0:
                frames.append(observe(env, gripper))
        gripper_start = gripper_goal

    return frames


def write_demo(path, frames, env, bddl_path, spec):
    path.parent.mkdir(parents=True, exist_ok=True)
    ee = np.asarray([frame["ee_pos"] for frame in frames])
    actions = np.zeros((len(frames), 4), dtype=np.float32)
    actions[:-1, :3] = np.diff(ee, axis=0)
    actions[:, 3] = [frame["gripper"] for frame in frames]
    success = bool(env.check_success())
    language = env.language_instruction

    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        data.attrs.update(
            num_demos=1,
            total=len(frames),
            action_format="delta_xyz_m_gripper",
            control_freq=RECORD_HZ,
            robot="NexArm",
            bddl_file=str(bddl_path),
            language=language,
        )
        demo = data.create_group("demo_0")
        demo.attrs.update(num_samples=len(frames), success=success)
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("task_language", data=language)
        demo.create_dataset("waypoints", data=json.dumps(spec))
        obs = demo.create_group("obs")
        for name in (
            "front_rgb",
            "wrist_rgb",
            "joint_pos",
            "joint_vel",
            "gripper_pos",
            "ee_pos",
            "proprio",
        ):
            values = np.asarray([frame[name] for frame in frames])
            compression = "gzip" if name.endswith("rgb") else None
            obs.create_dataset(name, data=values, compression=compression)
        rewards = np.zeros(len(frames), dtype=np.uint8)
        rewards[-1] = success
        demo.create_dataset("rewards", data=rewards)
        dones = np.zeros(len(frames), dtype=np.uint8)
        dones[-1] = 1
        demo.create_dataset("dones", data=dones)
    return success


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl", type=Path, required=True)
    parser.add_argument("--waypoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    spec = json.loads(args.waypoints.read_text())
    env = make_env(args.bddl)
    env.reset()
    settle(env)
    waypoints = resolve_waypoints(spec, env)
    joint_targets = solve_ik(env, [waypoint[0] for waypoint in waypoints])
    frames = execute(env, waypoints, joint_targets)
    success = write_demo(args.output, frames, env, args.bddl, spec)
    env.close()
    print(f"saved {len(frames)} samples, success={success}: {args.output}")


if __name__ == "__main__":
    main()
