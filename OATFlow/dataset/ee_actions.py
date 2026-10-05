"""Fixed-anchor world-frame EE delta of actuator command poses, with bounded IK."""
from __future__ import annotations

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from controllers.nexarm_mujoco_backend import JOINT_NAMES, MUJOCO_JOINTS
from controllers.oracle_pick import TCP_LOCAL
from OATFlow.environment.env import SCENE_XML, configure_mvp_model


def physical_joint_action(action, limits):
    action = np.asarray(action)
    fraction = np.concatenate(((action[..., :5] + 1) / 2, action[..., 5:6]), axis=-1)
    return limits[:, 0] + fraction * (limits[:, 1] - limits[:, 0])


class EEKinematics:
    def __init__(self, model=None):
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(SCENE_XML))
        if model is None:
            configure_mvp_model(self.model)
        self.data = mujoco.MjData(self.model)
        self.addresses = np.array([self.model.jnt_qposadr[self.model.joint(MUJOCO_JOINTS[name]).id]
                                   for name in JOINT_NAMES[:5]])
        self.body = self.model.body("link_6_gripper_base").id
        self.limits = self.model.actuator_ctrlrange.copy()

    def pose(self, joints):
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (5,) or not np.isfinite(joints).all():
            raise ValueError("EE FK requires five finite arm joints")
        self.data.qpos[self.addresses] = joints
        mujoco.mj_fwdPosition(self.model, self.data)
        rotation = self.data.xmat[self.body].reshape(3, 3).copy()
        return self.data.xpos[self.body].copy() + rotation @ TCP_LOCAL, rotation

    def inverse(self, position, rotation, initial, strict=True):
        low, high = self.limits[:5].T

        def residual(joints):
            point, matrix = self.pose(joints)
            return np.r_[100 * (point - position),
                         Rotation.from_matrix(matrix @ rotation.T).as_rotvec()]

        result = least_squares(residual, np.clip(initial, low + 1e-10, high - 1e-10),
                               bounds=(low, high), ftol=1e-11, xtol=1e-11, gtol=1e-11,
                               max_nfev=100)
        errors = residual(result.x)
        diagnostics = dict(position_error_m=float(np.linalg.norm(errors[:3]) / 100),
                           rotation_error_rad=float(np.linalg.norm(errors[3:])), nfev=result.nfev)
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError("EE IK did not converge")
        if strict and (diagnostics["position_error_m"] > 1e-5 or diagnostics["rotation_error_rad"] > 1e-4):
            raise ValueError(f"EE pose is unreachable: {diagnostics}")
        return result.x, diagnostics

    def decode(self, action, anchor, initial, strict=True):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("EE action requires XYZ, rotation vector, gripper")
        position, rotation = anchor
        desired_position = position + action[:3]
        desired_rotation = Rotation.from_rotvec(action[3:6]).as_matrix() @ rotation
        joints, diagnostics = self.inverse(desired_position, desired_rotation, initial, strict)
        gripper = np.clip(action[6], 0, 1)
        command = np.r_[joints, self.limits[5, 0] + gripper * np.diff(self.limits[5])[0]]
        return command.astype(np.float32), diagnostics


def ee_chunks(kinematics, action, state, valid, decision, name, discontinuities=()):
    """Encode actuator setpoint poses relative to each observed H25 chunk anchor."""
    from OATFlow.dataset.actions import chunk_starts

    starts, stop = chunk_starts(valid, decision, name, discontinuities)
    controls = physical_joint_action(action, kinematics.limits)
    observed = physical_joint_action(state, kinematics.limits)
    poses = [kinematics.pose(joints[:5]) for joints in controls]
    positions, rotations = np.stack([pose[0] for pose in poses]), np.stack([pose[1] for pose in poses])
    anchors = [kinematics.pose(observed[start, :5]) for start in starts]
    anchor_position = np.stack([pose[0] for pose in anchors])
    anchor_rotation = np.stack([pose[1] for pose in anchors])
    indices = starts[:, None] + np.arange(25)
    weights = indices < stop
    indices = np.minimum(indices, stop - 1)
    delta_position = positions[indices] - anchor_position[:, None]
    relative_rotation = rotations[indices] @ anchor_rotation[:, None].transpose(0, 1, 3, 2)
    delta_rotation = Rotation.from_matrix(relative_rotation.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 25, 3)
    labels = np.concatenate((delta_position, delta_rotation, action[indices, 5:6]), axis=-1).astype(np.float32)
    restored_rotation = Rotation.from_rotvec(labels[..., 3:6].reshape(-1, 3)).as_matrix().reshape(-1, 25, 3, 3)
    restored_rotation = restored_rotation @ anchor_rotation[:, None]
    if not np.allclose(labels[..., :3] + anchor_position[:, None], positions[indices], atol=1e-7, rtol=0) or \
            not np.allclose(restored_rotation, rotations[indices], atol=5e-7, rtol=0):
        raise ValueError(f"EE delta roundtrip failed: {name}")
    return dict(starts=starts, action=labels, valid=weights, joint_anchor=state[starts].copy(),
                anchor_position=anchor_position, anchor_rotation=anchor_rotation)
