"""Check URDF/MuJoCo frame agreement and the LIBERO-style EE delta action."""
from __future__ import annotations

import mujoco
import numpy as np

from controllers.nexarm_ik import NexArmIK
from envs.nexarm_env import NexArmEnv


def smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def main() -> None:
    env = NexArmEnv(
        scene_path="assets/robot/scene.xml",
        randomize_object=False,
        frame_skip=10,
    )

    ik = NexArmIK(
        urdf_path="assets/robot/robot.urdf",
        target_frame_name="link_6_gripper_base",
    )

    observation, _ = env.reset(seed=0)

    state = observation["observation.state"].astype(np.float64)
    current_arm_qpos = state[:5]

    # Kiểm tra URDF FK có khớp MuJoCo hay không.
    fk_pose = ik.forward_kinematics(current_arm_qpos)
    fk_xyz = fk_pose[:3, 3]

    body_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "link_6_gripper_base",
    )
    mujoco_xyz = env.data.xpos[body_id].copy()

    frame_error = np.linalg.norm(fk_xyz - mujoco_xyz)
    print("LeRobot FK XYZ:", fk_xyz)
    print("MuJoCo body XYZ:", mujoco_xyz)
    print("Frame error:", frame_error)

    if frame_error > 5e-3:
        raise RuntimeError(
            "URDF and MuJoCo frames differ by more than 5 mm. "
            "Fix joint/frame conventions before using EE delta control."
        )

    # Yêu cầu gripper-base đi lên 3 cm.
    target_xyz = fk_xyz + np.array([0.0, 0.0, 0.03])
    print("Target XYZ:", target_xyz)

    # Closed-loop Cartesian control through the LIBERO-style EE delta action.
    start_xyz = fk_xyz.copy()

    for step in range(150):
        alpha = smoothstep((step + 1) / 150)
        desired_xyz = start_xyz + alpha * (target_xyz - start_xyz)

        current_arm_qpos = observation["observation.state"][:5].astype(np.float64)
        current_xyz = ik.forward_kinematics(current_arm_qpos)[:3, 3]
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(
            (desired_xyz - current_xyz) / env.max_position_delta,
            -1.0,
            1.0,
        )
        action[6] = 1.0
        observation, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    final_arm_qpos = observation["observation.state"][:5].astype(np.float64)
    final_xyz = ik.forward_kinematics(final_arm_qpos)[:3, 3]
    print("Final arm qpos:     ", final_arm_qpos)
    print("Target XYZ:", target_xyz)
    print("Final XYZ (FK of actual):", final_xyz)
    print(
        "Cartesian error (target - actual FK):",
        float(np.linalg.norm(target_xyz - final_xyz)),
    )

    env.close()


if __name__ == "__main__":
    main()
