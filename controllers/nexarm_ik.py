from __future__ import annotations

from pathlib import Path

import numpy as np
from lerobot.model.kinematics import RobotKinematics


ARM_JOINT_NAMES = [
    "joint_1_base_to_link_1",
    "joint_2_link_1_to_link_2",
    "joint_3_link_2_to_link_3",
    "joint_4_link_3_to_link_4",
    "joint_5_link_4_to_link_5",
]


class NexArmIK:
    """LeRobot/Placo IK adapter for the MuJoCo NexArm environment.

    MuJoCo uses radians, while LeRobot RobotKinematics currently expects
    joint positions in degrees. This class performs the conversion.
    """

    def __init__(
        self,
        urdf_path: str | Path = "assets/robot/robot.urdf",
        target_frame_name: str = "link_6_gripper_base",
    ) -> None:
        urdf_path = Path(urdf_path).expanduser().resolve()

        if not urdf_path.exists():
            raise FileNotFoundError(urdf_path)

        self.kinematics = RobotKinematics(
            urdf_path=str(urdf_path),
            target_frame_name=target_frame_name,
            joint_names=ARM_JOINT_NAMES,
        )

    def forward_kinematics(
        self,
        arm_qpos_rad: np.ndarray,
    ) -> np.ndarray:
        arm_qpos_rad = np.asarray(arm_qpos_rad, dtype=np.float64)

        if arm_qpos_rad.shape != (5,):
            raise ValueError(
                f"Expected five arm joints, got {arm_qpos_rad.shape}"
            )

        return self.kinematics.forward_kinematics(
            np.rad2deg(arm_qpos_rad)
        )

    def solve_position(
        self,
        current_arm_qpos_rad: np.ndarray,
        target_xyz: np.ndarray,
    ) -> np.ndarray:
        current_arm_qpos_rad = np.asarray(
            current_arm_qpos_rad,
            dtype=np.float64,
        )
        target_xyz = np.asarray(target_xyz, dtype=np.float64)

        if current_arm_qpos_rad.shape != (5,):
            raise ValueError("current_arm_qpos_rad must have shape (5,)")

        if target_xyz.shape != (3,):
            raise ValueError("target_xyz must have shape (3,)")

        # Preserve the current orientation in the target transform.
        # orientation_weight=0 means this first test solves position only.
        desired_pose = self.forward_kinematics(
            current_arm_qpos_rad
        ).copy()
        desired_pose[:3, 3] = target_xyz

        target_deg = self.kinematics.inverse_kinematics(
            current_joint_pos=np.rad2deg(current_arm_qpos_rad),
            desired_ee_pose=desired_pose,
            position_weight=1.0,
            orientation_weight=0.0,
        )

        return np.deg2rad(
            np.asarray(target_deg[:5], dtype=np.float64)
        )