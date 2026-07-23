"""Test 2: a SO-101-style oracle grasps the cube through LeRobot's NexArm backend.

Faithful port of vla-so101/vla_data/oracle.py:

  * read cube ground-truth pose, plan targets as obj + offsets (no hardcoded
    waypoints);
  * fix the last two arm joints at GRASP_WRIST_RAD and solve 3-DOF POSITION-only
    IK for the first three joints (no orientation term — the wrist pose encodes the
    grasp orientation, exactly like SO-101's [26,-110]deg demo pose). The wrist
    stays at the base pose (j4=0, j5=0): no rotation needed;
  * closed-loop servo per step: cmd += clip(0.15*(target-actual), max_step) —
    single gain regime, matching SO-101;
  * grip is REAL jaw friction (no kinematic weld) — the corrected gripper_base
    collision box lets the jaws pinch the cube, which lifts and holds physically.

The NexArm-specific deviation from SO-101 is the grasp geometry: SO-101 descends
top-down (TCP above the cube); NexArm grasps from the SIDE — at the home pose the
jaws already point at the cube along local -y, so the wrist stays at the base pose
(j4=0, j5=0, NO rotation) and the cube enters along the jaw length axis. The grasp
target places the TCP SIDE_D in local +y from the cube so the cube sits at the jaw
tip. This is collision-honest once the gripper_base collision box is sized to the
real mesh (see assets/robot_ref/NexArm-sim.xml).

Control layer is LeRobot's `backend.step` (raw 0..4095). Records front+wrist MP4s.
Success: obj_z > 0.08 m held 10 steps.

Run: python scripts/test_oracle_grasp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.nexarm_mujoco_backend import JOINT_NAMES, NexArmMujocoBackend

FPS = 50
SITE_LOCAL = np.array([0.539369, -0.039412, 0.230434], dtype=np.float64)  # jaw centre in gripper_base frame
# SIDE grasp from the base pose: at home the jaws already point at the cube along
# local -y (jaw LENGTH axis), so NO wrist rotation is needed (j4=0, j5=0). The cube
# sits SIDE_D in local -y from the TCP. SIDE_D=0.018 seats it deep in the jaw (2.65cm
# of the 3cm cube inside the 5.8cm jaw, only 3.5mm protrudes past the tip) for a firm,
# natural grip; engage shove ~2.7mm, lift to ~11cm, hold above 0.08m (success).
# Residual ~3mm/s downward creep is MuJoCo soft-contact stick creep (wimpy spring
# gripper + asymmetric jaw actuation); firm close (kp=500) ejects the cube, stiff
# contact pops it out on lift, so the soft grip is the clean physical compromise.
GRASP_WRIST_RAD = np.deg2rad(np.array([0.0, 0.0], dtype=np.float64))
SIDE_D = 0.018  # cube centre this far in local -y from TCP (deep in jaw)
IK_TOL_M = 0.008
SERVO_K = 0.15
MAX_STEP = 0.015
# Gripper raw values (verified from jaw geometry, NOT assumption):
#   ctrl 0       -> raw 2833 -> left jaw 0.506, right jaw 0.573 -> gap 3.36 cm  = OPEN
#   ctrl -0.0255  -> raw 1195 -> left jaw 0.531, right jaw 0.548 -> jaws cross  = CLOSED
GRIPPER_OPEN_RAW = 2833.0
GRIPPER_CLOSED_RAW = 1195.0
LIFT_HEIGHT = 0.08
SUCCESS_HOLD_STEPS = 10
ARM_NAMES = list(JOINT_NAMES[:5])


def position_ik(backend, gripper_base_id, addrs, target, q0):
    """3-DOF position IK: fix the wrist at GRASP_WRIST_RAD, solve the first three
    joints for jaw-centre position. No orientation term (faithful to SO-101)."""
    model, data = backend.model, backend.data
    clone = mujoco.MjData(model)
    low3 = np.array([backend._control_range(n)[0] for n in ARM_NAMES[:3]], dtype=np.float64)
    high3 = np.array([backend._control_range(n)[1] for n in ARM_NAMES[:3]], dtype=np.float64)

    def fk(q5):
        clone.qpos[:] = data.qpos
        clone.qpos[addrs] = q5
        mujoco.mj_forward(model, clone)
        gpos = np.asarray(clone.xpos[gripper_base_id], dtype=np.float64).copy()
        gmat = np.asarray(clone.xmat[gripper_base_id], dtype=np.float64).reshape(3, 3).copy()
        return gpos + gmat @ SITE_LOCAL, gmat

    def residual(q3):
        tcp, _ = fk(np.concatenate([q3, GRASP_WRIST_RAD]))
        return 100.0 * (tcp - target)

    res = least_squares(residual, q0[:3], bounds=(low3, high3), max_nfev=400)
    q5 = np.concatenate([res.x, GRASP_WRIST_RAD])
    tcp, gmat = fk(q5)
    err = float(np.linalg.norm(tcp - target))
    if err > IK_TOL_M:
        raise RuntimeError(f"IK unreachable (pos {err:.4f} m) for target {np.round(target,4).tolist()}")
    return q5, gmat


def main() -> None:
    out = Path("outputs")
    out.mkdir(exist_ok=True)

    backend = NexArmMujocoBackend(
        model_path=Path("assets/robot_ref/scene.xml"),
        fps=FPS, camera_width=384, camera_height=384,
        camera_names=("front", "wrist"),
    )
    backend.reset(settle_steps=200)
    model, data = backend.model, backend.data

    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    gripper_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_6_gripper_base")
    if cube_body < 0 or gripper_base < 0:
        raise RuntimeError("reference scene must define body 'cube', body 'link_6_gripper_base'")
    addrs = np.array([model.jnt_qposadr[backend._joint_ids[n]] for n in ARM_NAMES], dtype=int)
    low = np.array([backend._control_range(n)[0] for n in ARM_NAMES], dtype=np.float64)
    high = np.array([backend._control_range(n)[1] for n in ARM_NAMES], dtype=np.float64)

    obj = np.asarray(data.xpos[cube_body], dtype=np.float64).copy()
    q0 = np.array([data.qpos[a] for a in addrs], dtype=np.float64).copy()
    print(f"object position: {np.round(obj,4).tolist()}")

    # Side grasp along the jaw length (local -y). Read the gripper local +y axis in
    # world at the grasp wrist via a throwaway IK to the cube, then place the TCP
    # SIDE_D in local +y from the cube (cube sits at the jaw tip in local -y).
    _, gmat_probe = position_ik(backend, gripper_base, addrs, obj, q0)
    local_y_world = gmat_probe[:, 1].copy()
    tcp_grasp = obj + SIDE_D * local_y_world
    tcp_above = tcp_grasp + 0.07 * local_y_world     # back off in local +y, in air
    tcp_above[2] = max(tcp_above[2], 0.10)            # keep the approach waypoint off the floor
    tcp_lift = tcp_grasp + np.array([0.0, 0.0, 0.10])

    above, _ = position_ik(backend, gripper_base, addrs, tcp_above, q0)
    grasp, gmat_g = position_ik(backend, gripper_base, addrs, tcp_grasp, above)
    lifted, _ = position_ik(backend, gripper_base, addrs, tcp_lift, grasp)
    approach = gmat_g[:, 2]
    print(f"  local+y (jaw length) = {np.round(local_y_world,3).tolist()}")
    print(f"  approach (local z)   = {np.round(approach,3).tolist()}")
    print(f"  tcp_above = {np.round(tcp_above,4).tolist()}")
    print(f"  tcp_grasp = {np.round(tcp_grasp,4).tolist()}")
    print(f"  tcp_lift  = {np.round(tcp_lift,4).tolist()}")
    print(f"  above   q={np.round(above,4).tolist()}")
    print(f"  grasp   q={np.round(grasp,4).tolist()}")
    print(f"  lifted  q={np.round(lifted,4).tolist()}")

    # No wrist rotation (j4=0, j5=0) -> the wrist is already at the grasp orientation
    # at home, so approach only needs to translate the arm in air; ~150 steps is
    # plenty at MAX_STEP 0.015 rad/step.
    stages = [
        ("1_approach", above,   GRIPPER_OPEN_RAW,   150),
        ("2_engage",   grasp,   GRIPPER_OPEN_RAW,   120),
        ("3_close",    grasp,   GRIPPER_CLOSED_RAW,  80),
        ("4_lift",     lifted,  GRIPPER_CLOSED_RAW, 160),
        ("5_hold",     lifted,  GRIPPER_CLOSED_RAW,  80),
    ]

    front = [backend.render("front")]
    wrist = [backend.render("wrist")]

    cmd = q0.copy()
    lift_steps = 0
    success = False
    reason = None

    step_idx = 0
    for name, q_target, g_raw, n_steps in stages:
        for _ in range(n_steps):
            actual = np.array([data.qpos[a] for a in addrs], dtype=np.float64)
            correction = np.clip(SERVO_K * (q_target - actual), -MAX_STEP, MAX_STEP)
            cmd = np.clip(cmd + correction, low, high)
            action = {f"{ARM_NAMES[j]}.pos": float(backend.control_to_raw(ARM_NAMES[j], float(cmd[j]))) for j in range(5)}
            action["gripper.pos"] = float(g_raw)

            if step_idx % 10 == 0 and name in ("2_engage", "3_close"):
                obj_pos = np.asarray(data.xpos[cube_body], dtype=np.float64)
                gmat_dyn = np.asarray(data.xmat[gripper_base], dtype=np.float64).reshape(3, 3)
                j4_act = float(np.rad2deg(data.qpos[addrs[3]]))
                j5_act = float(np.rad2deg(data.qpos[addrs[4]]))
                cubeg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_collision")
                n_arm_cube = sum(1 for ci in range(data.ncon) if cubeg in {data.contact[ci].geom1, data.contact[ci].geom2})
                print(f"    [{name} s{step_idx}] j4={j4_act:7.1f} j5={j5_act:7.1f} appr={np.round(gmat_dyn[:,2],3).tolist()} obj={np.round(obj_pos,4).tolist()} jawcube_contacts={n_arm_cube}")

            backend.step(action)
            front.append(backend.render("front"))
            wrist.append(backend.render("wrist"))

            z = float(data.xpos[cube_body][2])
            lift_steps = lift_steps + 1 if z > LIFT_HEIGHT else 0
            if lift_steps >= SUCCESS_HOLD_STEPS:
                success, reason = True, "success"
                break
            step_idx += 1
        print(f"Phase {name}: obj_z={float(data.xpos[cube_body][2]):.4f} success={success}")
        if reason is not None:
            break

    iio.imwrite(out / "oracle_front.mp4", np.array(front), fps=FPS)
    iio.imwrite(out / "oracle_wrist.mp4", np.array(wrist), fps=FPS)
    backend.close()
    print(f"\nframes: {len(front)}")
    print(f"success: {success}")
    print(f"reason: {reason}")
    print(f"final obj_z: {float(data.xpos[cube_body][2]):.4f} m")
    print(f"front: {out/'oracle_front.mp4'}")
    print(f"wrist: {out/'oracle_wrist.mp4'}")


if __name__ == "__main__":
    main()