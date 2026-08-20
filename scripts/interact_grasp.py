"""Interactive MuJoCo viewer for the side grasp: opens launch_passive on the same
model/data the backend steps, runs the grasp sequence in real time so you can
orbit the camera and watch the jaws close on the cube. After the sequence it
keeps the viewer open until you close the window.

Run: python scripts/interact_grasp.py
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from controllers.nexarm_mujoco_backend import JOINT_NAMES, NexArmMujocoBackend

FPS = 50
SITE_LOCAL = np.array([0.539369, -0.039412, 0.230434], dtype=np.float64)
GRASP_WRIST_RAD = np.deg2rad(np.array([0.0, 0.0], dtype=np.float64))
SIDE_D = 0.018
IK_TOL_M = 0.008
SERVO_K = 0.15
MAX_STEP = 0.015
GRIPPER_OPEN_RAW = 2833.0
GRIPPER_CLOSED_RAW = 1195.0
ARM_NAMES = list(JOINT_NAMES[:5])


def position_ik(backend, gb_id, addrs, target, q0):
    model, data = backend.model, backend.data
    clone = mujoco.MjData(model)
    low3 = np.array([backend._control_range(n)[0] for n in ARM_NAMES[:3]], dtype=np.float64)
    high3 = np.array([backend._control_range(n)[1] for n in ARM_NAMES[:3]], dtype=np.float64)

    def fk(q5):
        clone.qpos[:] = data.qpos
        clone.qpos[addrs] = q5
        mujoco.mj_forward(model, clone)
        gp = np.asarray(clone.xpos[gb_id], dtype=np.float64).copy()
        gm = np.asarray(clone.xmat[gb_id], dtype=np.float64).reshape(3, 3).copy()
        return gp + gm @ SITE_LOCAL, gm

    def residual(q3):
        tcp, _ = fk(np.concatenate([q3, GRASP_WRIST_RAD]))
        return 100.0 * (tcp - target)

    res = least_squares(residual, q0[:3], bounds=(low3, high3), max_nfev=400)
    q5 = np.concatenate([res.x, GRASP_WRIST_RAD])
    tcp, gm = fk(q5)
    err = float(np.linalg.norm(tcp - target))
    if err > IK_TOL_M:
        raise RuntimeError(f"IK unreachable {err:.4f}")
    return q5, gm


def main() -> None:
    b = NexArmMujocoBackend(model_path=Path("assets/robot/scene.xml"),
                            fps=FPS, camera_width=160, camera_height=160, camera_names=("front",))
    b.reset(settle_steps=200)
    m, d = b.model, b.data
    cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
    gb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link_6_gripper_base")
    addrs = np.array([m.jnt_qposadr[b._joint_ids[n]] for n in ARM_NAMES], dtype=int)
    low = np.array([b._control_range(n)[0] for n in ARM_NAMES], dtype=np.float64)
    high = np.array([b._control_range(n)[1] for n in ARM_NAMES], dtype=np.float64)
    obj = np.asarray(d.xpos[cube_body], dtype=np.float64).copy()
    q0 = np.array([d.qpos[a] for a in addrs], dtype=np.float64).copy()

    _, gm_probe = position_ik(b, gb, addrs, obj, q0)
    ly = gm_probe[:, 1].copy()
    tcp_grasp = obj + SIDE_D * ly
    tcp_above = tcp_grasp + 0.07 * ly
    tcp_above[2] = max(tcp_above[2], 0.10)
    tcp_lift = tcp_grasp + np.array([0.0, 0.0, 0.10])
    above, _ = position_ik(b, gb, addrs, tcp_above, q0)
    grasp, _ = position_ik(b, gb, addrs, tcp_grasp, above)
    lifted, _ = position_ik(b, gb, addrs, tcp_lift, grasp)

    stages = [("1_approach", above, GRIPPER_OPEN_RAW, 150),
              ("2_engage", grasp, GRIPPER_OPEN_RAW, 120),
              ("3_close", grasp, GRIPPER_CLOSED_RAW, 80),
              ("4_lift", lifted, GRIPPER_CLOSED_RAW, 160),
              ("5_hold", lifted, GRIPPER_CLOSED_RAW, 80)]

    cmd = q0.copy()
    dt = 1.0 / FPS

    with mujoco.viewer.launch_passive(m, d) as viewer:
        print("viewer open. Grasping in real time — orbit with mouse. Close window to quit.")
        for name, qt, graw, ns in stages:
            for _ in range(ns):
                if not viewer.is_running():
                    return
                actual = np.array([d.qpos[a] for a in addrs], dtype=np.float64)
                cmd = np.clip(cmd + np.clip(SERVO_K * (qt - actual), -MAX_STEP, MAX_STEP), low, high)
                action = {f"{ARM_NAMES[j]}.pos": float(b.control_to_raw(ARM_NAMES[j], float(cmd[j]))) for j in range(5)}
                action["gripper.pos"] = float(graw)
                b.step(action)
                viewer.sync()
                time.sleep(dt)
            print(f"  {name}: obj_z={float(d.xpos[cube_body][2]):.4f}")
        print("sequence done — viewer stays open. Close the window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(dt)
    b.close()


if __name__ == "__main__":
    main()
