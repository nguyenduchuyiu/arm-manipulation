"""Forward-kinematics grasp driven by LeRobot's NexArm control backend.

Control layer: the vendored `NexArmMujocoBackend` (LeRobot's ready-made control
code for this arm, PR huggingface/lerobot#3972). Its contract is the SO-101
follower path: `step(action)` takes raw servo positions 0..4095, linear-maps them
to the MuJoCo position actuators (the firmware analogue — no hand-rolled cosine
servo), and runs `mj_step(steps_per_action)`. We feed it a smooth reference
trajectory (a stream of `send_action` calls, exactly as leader teleop does).

Planning layer: FK-search (unchanged) — random-restart + hill-climb over the arm
actuator range, scoring the SETTLED grasp_site in a sim clone, because position
actuators sag under gravity so FK(q) != held pose.

Task layer: kinematic snap-grasp (weld cube to gripper base frame on jaw-close),
the documented-missing piece in the lerobot-nexarm reference (its README lists
grasp-lift success as remaining work). Success: cube centre held > 0.08 m for
SUCCESS_HOLD_STEPS control steps.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.nexarm_ik import NexArmIK
from controllers.nexarm_mujoco_backend import (
    HOME_POSITIONS,
    JOINT_NAMES,
    RAW_RANGES,
    NexArmMujocoBackend,
)


SITE_LOCAL = np.array([0.539369, -0.039412, 0.230434], dtype=np.float64)
FK_TOL_M = 0.025
RANDOM_RESTARTS = 200
SETTLE_SIM_STEPS = 1200
APPROACH_AXIS = np.array([0.0, 0.0, -1.0], dtype=np.float64)
GAP_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)
ORIENT_W = 0.02
RAMP_STEPS = 30
HOLD_STEPS = 150
GRIPPER_OPEN_RAW = 1195.0      # RAW_RANGES["gripper"] low  -> ctrl -0.0255 (open)
GRIPPER_CLOSED_RAW = 2833.0    # RAW_RANGES["gripper"] high -> ctrl 0 (closed)
SNAP_DISTANCE = 0.06
LIFT_HEIGHT = 0.08
SUCCESS_HOLD_STEPS = 10
FPS = 50


def _grasp_site(ik: NexArmIK, q5: np.ndarray) -> np.ndarray:
    pose = ik.forward_kinematics(q5)
    return pose[:3, 3] + pose[:3, :3] @ SITE_LOCAL


def _orient_err(R: np.ndarray) -> float:
    approach = R[:, 2]
    gap = R[:, 0]
    return float((1.0 - approach @ APPROACH_AXIS) + (1.0 - abs(gap @ GAP_AXIS)))


def _settle_pose(ik, model, clone, addrs, q_cmd: np.ndarray, snap: np.ndarray):
    clone.qpos[:] = snap
    clone.qvel[:] = 0.0
    clone.ctrl[:5] = q_cmd
    clone.ctrl[5] = -0.0255  # gripper open during search
    mujoco.mj_forward(model, clone)
    for _ in range(SETTLE_SIM_STEPS):
        mujoco.mj_step(model, clone)
    pose = ik.forward_kinematics(clone.qpos[addrs].copy())
    site = pose[:3, 3] + pose[:3, :3] @ SITE_LOCAL
    return site, pose[:3, :3]


def _score(ik, model, clone, addrs, q_cmd, snap, target_xyz, orient_w):
    site, R = _settle_pose(ik, model, clone, addrs, q_cmd, snap)
    pos_err = float(np.linalg.norm(site - target_xyz))
    return pos_err + orient_w * _orient_err(R), pos_err


def fk_search(ik, model, clone, addrs, rng, target_xyz, snap, q0,
              restarts=RANDOM_RESTARTS, orient_w=ORIENT_W):
    low = model.actuator_ctrlrange[:5, 0].astype(np.float64)
    high = model.actuator_ctrlrange[:5, 1].astype(np.float64)
    q = np.clip(q0, low, high)
    best_total, best_pos = _score(ik, model, clone, addrs, q, snap, target_xyz, orient_w)
    for _ in range(restarts):
        sample = rng.uniform(low, high)
        total, pos = _score(ik, model, clone, addrs, sample, snap, target_xyz, orient_w)
        if total < best_total:
            best_total, best_pos, q = total, pos, sample.copy()
    step = 0.1
    while best_pos > FK_TOL_M and step > 1e-3:
        improved = False
        for j in range(5):
            for d in (-step, step):
                cand = q.copy()
                cand[j] = float(np.clip(q[j] + d, low[j], high[j]))
                if cand[j] == q[j]:
                    continue
                total, pos = _score(ik, model, clone, addrs, cand, snap, target_xyz, orient_w)
                if total < best_total:
                    best_total, best_pos, q, improved = total, pos, cand, True
        if not improved:
            step /= 2.0
    if best_pos > FK_TOL_M:
        raise RuntimeError(f"FK search did not reach target (error {best_pos:.4f} m)")
    return q


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="assets/robot_ref/scene.xml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True)

    backend = NexArmMujocoBackend(
        model_path=Path(args.scene),
        fps=FPS,
        camera_width=384,
        camera_height=384,
        camera_names=("front", "wrist"),
    )
    backend.reset(settle_steps=200)

    model = backend.model
    data = backend.data

    # Object + gripper frame handles for the snap-grasp task layer.
    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
    gripper_base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_6_gripper_base")
    if cube_body < 0 or cube_joint < 0 or gripper_base_body < 0:
        raise RuntimeError("reference scene must define body 'cube', joint 'cube_joint', body 'link_6_gripper_base'")
    cube_qadr = int(model.jnt_qposadr[cube_joint])

    # Arm-joint qpos addresses (joints 1..5) for FK-search clone settling.
    arm_feature_names = JOINT_NAMES[:5]
    addrs = np.array([model.jnt_qposadr[backend._joint_ids[n]] for n in arm_feature_names], dtype=int)

    ik = NexArmIK()
    rng = np.random.default_rng(args.seed)
    clone = mujoco.MjData(model)

    obj = np.asarray(data.xpos[cube_body], dtype=np.float64).copy()
    print(f"object position: {obj.tolist()}")

    # Frame check: URDF FK link_6_gripper_base must match the reference MuJoCo
    # body xpos, else snap-detection (URDF grasp_site) and the weld (MuJoCo
    # xpos) disagree.
    q5_home = np.array([data.qpos[addrs[j]] for j in range(5)], dtype=np.float64)
    pose_home = ik.forward_kinematics(q5_home)
    urdf_base = pose_home[:3, 3]
    mj_base = np.asarray(data.xpos[gripper_base_body], dtype=np.float64)
    print(f"frame check: URDF link_6_gripper_base={np.round(urdf_base,4).tolist()} "
          f"MuJoCo xpos={np.round(mj_base,4).tolist()} "
          f"diff={np.linalg.norm(urdf_base-mj_base):.4f}")

    def plan_phase(target_offset, q0, label, orient_w=ORIENT_W):
        snap = data.qpos.copy()
        target = obj + np.array(target_offset, dtype=np.float64)
        q = fk_search(ik, model, clone, addrs, rng, target, snap, q0, orient_w=orient_w)
        site = _grasp_site(ik, q)
        print(f"  {label}: cmd(rad)={np.round(q,4).tolist()} static_site={np.round(site,4).tolist()} "
              f"target={np.round(target,4).tolist()}")
        return q  # rad (control space)

    q0 = np.zeros(5, dtype=np.float64)
    grasp = plan_phase([0.0, 0.0, 0.0], q0, "grasp")
    lift = plan_phase([0.0, 0.13, 0.08], grasp, "lift", orient_w=0.0)

    # Convert rad plan targets to raw 0..4095 for the LeRobot action contract.
    def to_raw(q_rad):
        return np.array([backend.control_to_raw(arm_feature_names[j], float(q_rad[j])) for j in range(5)])

    grasp_raw = to_raw(grasp)
    lift_raw = to_raw(lift)

    plan = [
        (grasp_raw, GRIPPER_OPEN_RAW, "1_approach", False),
        (grasp_raw, GRIPPER_CLOSED_RAW, "2_grasp", True),
        (lift_raw, GRIPPER_CLOSED_RAW, "3_lifting", True),
        (lift_raw, GRIPPER_CLOSED_RAW, "4_lifted", True),
    ]

    front_frames = [backend.render("front")]
    wrist_frames = [backend.render("wrist")]
    snapshots = {}

    grasped = False
    grasp_local_offset = None
    lift_steps = 0
    success = False
    terminated_reason = None

    def object_z():
        return float(data.xpos[cube_body][2])

    def weld_object():
        gpos = np.asarray(data.xpos[gripper_base_body], dtype=np.float64)
        gmat = np.asarray(data.xmat[gripper_base_body], dtype=np.float64).reshape(3, 3)
        world = gpos + gmat @ grasp_local_offset
        data.qpos[cube_qadr:cube_qadr + 3] = world
        data.qvel[cube_qadr:cube_qadr + 3] = 0.0
        data.qvel[cube_qadr + 3:cube_qadr + 6] = 0.0

    # Start from the settled home joint positions (raw) for a smooth ramp origin.
    curr = backend.joint_positions()
    curr_q = np.array([curr[f"{n}.pos"] for n in arm_feature_names], dtype=np.float64)
    curr_g = float(curr["gripper.pos"])

    max_steps = (RAMP_STEPS + HOLD_STEPS) * len(plan) + 50
    step_idx = 0

    for p_idx, (t_q, t_g, snap_key, intends_closed) in enumerate(plan):
        s_q, s_g = curr_q.copy(), curr_g
        for s in range(RAMP_STEPS + HOLD_STEPS):
            if s < RAMP_STEPS:
                t = 0.5 * (1.0 - np.cos((s + 1) / float(RAMP_STEPS) * np.pi))
                q_cmd = (1.0 - t) * s_q + t * t_q
                g_cmd = (1.0 - t) * s_g + t * t_g
            else:
                q_cmd, g_cmd = t_q, t_g
            action = {f"{arm_feature_names[j]}.pos": float(q_cmd[j]) for j in range(5)}
            action["gripper.pos"] = float(g_cmd)

            # Snap-grasp: trigger when this phase intends closed and the cube is
            # near the TCP; hold for the whole closed phase. Release is driven by
            # phase intent, NOT the raw jaw command — the cube (3.6 cm) is wider
            # than the jaw opening (2.55 cm), so the jaw is physically blocked
            # open at raw ~1677 even when closed is commanded; an instantaneous
            # threshold would release the cube the moment the next phase's ramp
            # starts from that blocked position.
            obj_pos = np.asarray(data.xpos[cube_body], dtype=np.float64)
            q5_now = np.array([data.qpos[addrs[j]] for j in range(5)], dtype=np.float64)
            grasp_pos = _grasp_site(ik, q5_now)
            dist = float(np.linalg.norm(obj_pos - grasp_pos))
            if intends_closed and not grasped and dist < SNAP_DISTANCE:
                grasped = True
                gpos = np.asarray(data.xpos[gripper_base_body], dtype=np.float64)
                gmat = np.asarray(data.xmat[gripper_base_body], dtype=np.float64).reshape(3, 3)
                grasp_local_offset = gmat.T @ (obj_pos - gpos)
                print(f"  SNAP at step {step_idx}: dist={dist:.4f} obj={np.round(obj_pos,4).tolist()} "
                      f"gripper_base_z={float(data.xpos[gripper_base_body][2]):.4f}")
            elif not intends_closed and grasped:
                grasped = False
                grasp_local_offset = None

            if grasped:
                weld_object()

            backend.step(action)  # LeRobot control: raw -> actuators -> mj_step

            if grasped:
                weld_object()

            front_frames.append(backend.render("front"))
            wrist_frames.append(backend.render("wrist"))

            z = object_z()
            if z > LIFT_HEIGHT:
                lift_steps += 1
            else:
                lift_steps = 0
            if lift_steps >= SUCCESS_HOLD_STEPS:
                success = True
                terminated_reason = "success"
                break

            step_idx += 1
            if step_idx >= max_steps:
                terminated_reason = "timeout"
                break

        curr = backend.joint_positions()
        curr_q = np.array([curr[f"{n}.pos"] for n in arm_feature_names], dtype=np.float64)
        curr_g = float(curr["gripper.pos"])
        snapshots[f"fk_{snap_key}_front.png"] = backend.render("front")
        snapshots[f"fk_{snap_key}_wrist.png"] = backend.render("wrist")
        print(f"Phase {p_idx+1} ({snap_key}): obj_z={object_z():.4f} m, success={success}")
        if terminated_reason is not None:
            break

    fps = FPS
    front_path = output_dir / "fk_grasp_front.mp4"
    wrist_path = output_dir / "fk_grasp_wrist.mp4"
    iio.imwrite(front_path, np.array(front_frames), fps=fps)
    iio.imwrite(wrist_path, np.array(wrist_frames), fps=fps)
    for name, img in snapshots.items():
        iio.imwrite(output_dir / name, img)
    backend.close()

    print(f"\nframes: {len(front_frames)}")
    print(f"success: {success}")
    print(f"terminated_reason: {terminated_reason}")
    print(f"final object z: {object_z():.4f} m")
    print(f"front mp4: {front_path}")
    print(f"wrist mp4: {wrist_path}")


if __name__ == "__main__":
    main()