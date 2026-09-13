from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


ARM_JOINT_NAMES = [
    "joint_1_base_to_link_1",
    "joint_2_link_1_to_link_2",
    "joint_3_link_2_to_link_3",
    "joint_4_link_3_to_link_4",
    "joint_5_link_4_to_link_5",
]


class NexArmIK:
    """Forward and inverse kinematics computed from the MuJoCo model."""

    def __init__(
        self,
        model_path: str | Path = "assets/robot/robot.xml",
        target_frame_name: str = "link_6_gripper_base",
        *,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        damping: float = 1e-4,
    ) -> None:
        model_path = Path(model_path).expanduser()
        if not model_path.is_absolute():
            if model_path.is_file():
                model_path = model_path.resolve()
            else:
                model_path = Path(__file__).resolve().parents[1] / model_path
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.damping = damping

        self._body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, target_frame_name
        )
        if self._body_id < 0:
            raise ValueError(
                f"MuJoCo model is missing target body {target_frame_name!r}"
            )

        joint_ids = []
        for name in ARM_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError(f"MuJoCo model is missing arm joint {name!r}")
            joint_ids.append(joint_id)

        self._joint_ids = np.asarray(joint_ids, dtype=int)
        self._qpos_addrs = self.model.jnt_qposadr[self._joint_ids].astype(int)
        self._dof_addrs = self.model.jnt_dofadr[self._joint_ids].astype(int)
        self._joint_low = self.model.jnt_range[self._joint_ids, 0].copy()
        self._joint_high = self.model.jnt_range[self._joint_ids, 1].copy()

    def _set_arm_qpos(self, arm_qpos_rad: np.ndarray) -> None:
        arm_qpos_rad = np.asarray(arm_qpos_rad, dtype=np.float64)
        if arm_qpos_rad.shape != (5,):
            raise ValueError(f"Expected five arm joints, got {arm_qpos_rad.shape}")
        self.data.qpos[self._qpos_addrs] = arm_qpos_rad
        mujoco.mj_forward(self.model, self.data)

    def forward_kinematics(self, arm_qpos_rad: np.ndarray) -> np.ndarray:
        self._set_arm_qpos(arm_qpos_rad)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.data.xmat[self._body_id].reshape(3, 3)
        pose[:3, 3] = self.data.xpos[self._body_id]
        return pose

    def solve_position(
        self,
        current_arm_qpos_rad: np.ndarray,
        target_xyz: np.ndarray,
    ) -> np.ndarray:
        qpos = np.asarray(current_arm_qpos_rad, dtype=np.float64).copy()
        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        if qpos.shape != (5,):
            raise ValueError("current_arm_qpos_rad must have shape (5,)")
        if target_xyz.shape != (3,):
            raise ValueError("target_xyz must have shape (3,)")

        jacobian = np.zeros((3, self.model.nv), dtype=np.float64)
        identity = np.eye(3, dtype=np.float64)
        for _ in range(self.max_iterations):
            self._set_arm_qpos(qpos)
            error = target_xyz - self.data.xpos[self._body_id]
            if np.linalg.norm(error) <= self.tolerance:
                break

            jacobian.fill(0.0)
            mujoco.mj_jacBody(
                self.model, self.data, jacobian, None, self._body_id
            )
            arm_jacobian = jacobian[:, self._dof_addrs]
            step = arm_jacobian.T @ np.linalg.solve(
                arm_jacobian @ arm_jacobian.T + self.damping * identity,
                error,
            )
            qpos = np.clip(qpos + step, self._joint_low, self._joint_high)

        return qpos
