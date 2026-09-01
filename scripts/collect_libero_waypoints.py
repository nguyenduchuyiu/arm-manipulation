"""Collect a NexArm demonstration from Cartesian waypoints in an unchanged LIBERO task."""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
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
OBJECT_SCALE = 0.35


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


def collision_proxies(env):
    model, data = env.sim.model, env.sim.data
    proxies = {}
    signs = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    )
    for obj in env.env.objects_dict.values():
        root_id = model.body_name2id(obj.root_body)
        root_pos = data.body_xpos[root_id]
        root_rot = data.body_xmat[root_id].reshape(3, 3)
        points = []
        for name in obj.contact_geoms:
            geom_id = model.geom_name2id(name)
            geom_type = model.geom_type[geom_id]
            size = model.geom_size[geom_id]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                extent = size
            elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                extent = size
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                extent = np.repeat(size[0], 3)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                extent = np.array([size[0], size[0], size[1]])
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                extent = np.array([size[0], size[0], size[0] + size[1]])
            else:
                extent = np.repeat(model.geom_rbound[geom_id], 3)
            geom_rot = data.geom_xmat[geom_id].reshape(3, 3)
            corners = data.geom_xpos[geom_id] + (signs * extent) @ geom_rot.T
            points.append((corners - root_pos) @ root_rot)
        points = np.concatenate(points)
        lower, upper = points.min(axis=0), points.max(axis=0)
        proxies[obj.root_body] = (
            obj.contact_geoms,
            0.5 * (lower + upper),
            0.5 * (upper - lower),
        )
    return proxies


def scale_movable_objects(env):
    proxies = collision_proxies(env)
    old_model = env.sim.model._model
    old_qpos = env.sim.data.qpos.copy()
    joint_qpos = {}
    for joint_id in range(old_model.njnt):
        name = mujoco.mj_id2name(old_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        start = old_model.jnt_qposadr[joint_id]
        end = old_model.jnt_qposadr[joint_id + 1] if joint_id + 1 < old_model.njnt else old_model.nq
        joint_qpos[name] = old_qpos[start:end].copy()
    root = ET.fromstring(env.env.model.get_xml())
    parents = {child: parent for parent in root.iter() for child in parent}
    mesh_names = set()
    for obj in env.env.objects_dict.values():
        body = root.find(f".//body[@name='{obj.root_body}']")
        if body is None:
            raise RuntimeError(f"missing object body in MJCF: {obj.root_body}")
        for element in body.iter():
            if element.get("mesh"):
                mesh_names.add(element.get("mesh"))
            for attribute in ("pos", "size", "fromto"):
                if attribute not in element.attrib or (element is body and attribute == "pos"):
                    continue
                values = np.fromstring(element.get(attribute), sep=" ") * OBJECT_SCALE
                element.set(attribute, " ".join(map(str, values)))
            if "mass" in element.attrib:
                element.set("mass", str(float(element.get("mass")) * OBJECT_SCALE))
            for attribute in ("diaginertia", "fullinertia"):
                if attribute in element.attrib:
                    values = np.fromstring(element.get(attribute), sep=" ") * OBJECT_SCALE**3
                    element.set(attribute, " ".join(map(str, values)))

        contact_names, center, half_size = proxies[obj.root_body]
        for name in contact_names:
            geom = root.find(f".//geom[@name='{name}']")
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        proxy = root.find(f".//geom[@name='{contact_names[0]}']")
        parents[proxy].remove(proxy)
        body.append(
            ET.Element(
                "geom",
                name=contact_names[0],
                type="box",
                pos=" ".join(map(str, center * OBJECT_SCALE)),
                size=" ".join(map(str, half_size * OBJECT_SCALE)),
                group="0",
                friction="0.95 0.01 0.001",
                solref="0.01 1",
                solimp="0.9 0.95 0.001 0.5 2",
            )
        )
    for mesh in root.findall("./asset/mesh"):
        if mesh.get("name") in mesh_names:
            values = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            mesh.set("scale", " ".join(map(str, values * OBJECT_SCALE)))

    env.reset_from_xml_string(ET.tostring(root, encoding="unicode"))
    model = env.sim.model._model
    for name, values in joint_qpos.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"joint disappeared after object scaling: {name}")
        start = model.jnt_qposadr[joint_id]
        env.sim.data.qpos[start : start + len(values)] = values
    for obj in env.env.objects_dict.values():
        for joint_name in obj.joints:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                env.sim.data.qpos[model.jnt_qposadr[joint_id] + 2] += 0.003
    env.sim.data.qvel[:] = 0.0
    env.sim.forward()


def oracle_waypoints(env):
    goals = env.env.parsed_problem["goal_state"]
    if len(goals) != 1 or goals[0][0] != "on":
        raise RuntimeError("oracle currently supports one On(object, target) goal")
    _, object_name, target_name = goals[0]
    object_pos = env.sim.data.body_xpos[env.env.obj_body_id[object_name]].copy()
    target_pos = env.sim.data.body_xpos[env.env.obj_body_id[target_name]].copy()
    above = np.array([0.0, 0.0, 0.12])
    grasp = np.array([0.0, 0.0, 0.025])
    place = np.array([0.0, 0.0, 0.035])
    waypoints = [
        (object_pos + above, 1.0, 1.0),
        (object_pos + grasp, 1.0, 0.8),
        (object_pos + grasp, -1.0, 0.5),
        (object_pos + above, -1.0, 0.8),
        (target_pos + above, -1.0, 1.0),
        (target_pos + place, -1.0, 0.8),
        (target_pos + place, 1.0, 0.5),
        (target_pos + above, 1.0, 0.8),
    ]
    spec = {
        "oracle": "on_object_target",
        "object": object_name,
        "target": target_name,
        "object_scale": OBJECT_SCALE,
        "waypoints": [xyz.tolist() for xyz, _, _ in waypoints],
    }
    return waypoints, spec


def place_robot(env, waypoints):
    pick, target = waypoints[1][0], waypoints[5][0]
    delta = target[:2] - pick[:2]
    perpendicular = np.array([-delta[1], delta[0]])
    perpendicular /= max(np.linalg.norm(perpendicular), 1e-6)
    if perpendicular[1] < 0:
        perpendicular *= -1
    shoulder = 0.5 * (pick[:2] + target[:2]) + 0.10 * perpendicular
    robot = env.robots[0]
    root_id = env.sim.model.body_name2id(robot.robot_model.root_body)
    current_shoulder = env.sim.data.body_xpos[root_id, :2] + np.array([0.539369, 0.063973])
    env.sim.model.body_pos[root_id, :2] += shoulder - current_shoulder
    env.sim.forward()
    robot.controller.update_base_pose(robot.base_pos, robot.base_ori)


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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_env(args.bddl)
    env.reset()
    settle(env)
    scale_movable_objects(env)
    settle(env)
    waypoints, spec = oracle_waypoints(env)
    place_robot(env, waypoints)
    joint_targets = solve_ik(env, [waypoint[0] for waypoint in waypoints])
    frames = execute(env, waypoints, joint_targets)
    success = write_demo(args.output, frames, env, args.bddl, spec)
    env.close()
    print(f"saved {len(frames)} samples, success={success}: {args.output}")


if __name__ == "__main__":
    main()
