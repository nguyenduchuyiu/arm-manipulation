"""World-axis pose deltas and a bounded five-joint pose IK controller."""
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ARM_JOINTS = ['joint_1_base_to_link_1', 'joint_2_link_1_to_link_2',
              'joint_3_link_2_to_link_3', 'joint_4_link_3_to_link_4', 'joint_5_link_4_to_link_5']
ACTION_DIM = 10
REPRESENTATION = 'ee_pose_delta_anchor'
UNITS = ['m'] * 3 + ['unitless'] * 6 + ['m']
STATE_DIM = 10
STATE_REPRESENTATION = 'ee_position_rotation6d_gripper_base'
GRIPPER_TRAVEL_M = .0255
GRIPPER_NORMALIZATION = 'fixed_range_open_minus1_closed_plus1'


def normalize_gripper(position_m):
    return -1 - 2 * position_m / GRIPPER_TRAVEL_M


def denormalize_gripper(value):
    return -GRIPPER_TRAVEL_M * (value + 1) / 2


def rotation_6d(matrix):
    # First two columns, stored column by column.
    return np.swapaxes(matrix[..., :, :2], -1, -2).reshape(*matrix.shape[:-2], 6)


def base_state(position_world, rotation_world, base_origin_world, gripper_position):
    """The workspace base axes match world; only the origin is translated."""
    return np.concatenate((np.asarray(position_world) - base_origin_world,
                           rotation_6d(np.asarray(rotation_world)),
                           np.asarray(gripper_position)[..., None]), axis=-1)


def rotation_matrix(value):
    a, b = np.asarray(value, dtype=np.float64)[..., :3], np.asarray(value, dtype=np.float64)[..., 3:]
    x = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    y = b - (x * b).sum(-1, keepdims=True) * x
    if np.any(np.linalg.norm(a, axis=-1) < 1e-8) or np.any(np.linalg.norm(y, axis=-1) < 1e-8):
        raise ValueError('Degenerate rotation 6D prediction')
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def measured_rotations(model, qpos, expected_ee):
    state = mujoco.MjData(model)
    result = np.empty((len(qpos), 3, 3))
    positions = np.empty((len(qpos), 3))
    site = model.site('grasp_site').id
    for i in range(len(qpos)):
        state.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, state)
        positions[i] = state.site_xpos[site]
        result[i] = state.site_xmat[site].reshape(3, 3)
    np.testing.assert_allclose(positions, expected_ee, rtol=0, atol=6e-8)
    return result


def pose_labels(ee, rotations, commands, starts, horizon=25):
    total = len(commands)
    indices = 2 * (starts[:, None] + np.arange(horizon))
    future = np.minimum(indices + 2, total)
    anchor_p, anchor_r = ee[2 * starts], rotations[2 * starts]
    action = np.empty((len(starts), horizon, ACTION_DIM), np.float32)
    action[..., :3] = ee[future] - anchor_p[:, None]
    action[..., 3:9] = rotation_6d(rotations[future] @ np.swapaxes(anchor_r[:, None], -1, -2))
    action[..., 9] = commands[np.minimum(indices, 2 * ((total - 1) // 2)), 5]
    mask = indices < total
    np.testing.assert_allclose(anchor_p[:, None] + action[..., :3], ee[future], rtol=0, atol=6e-8)
    np.testing.assert_allclose(rotation_matrix(action[..., 3:9]) @ anchor_r[:, None],
                               rotations[future], rtol=0, atol=2e-7)
    assert np.isfinite(action).all()
    for values, valid in zip(action, mask):
        length = int(valid.sum())
        np.testing.assert_array_equal(values[length:], np.broadcast_to(values[length - 1], (horizon - length, ACTION_DIM)))
    return action, mask, anchor_p, anchor_r, future


def angle_error(rotation, target):
    return np.linalg.norm(Rotation.from_matrix(rotation @ np.swapaxes(target, -1, -2)).as_rotvec(), axis=-1)


class PoseIK:
    def __init__(self, physics):
        self.model, self.qadr, self.dofs, self.site = physics
        self.state = mujoco.MjData(self.model)
        self.low, self.high = self.model.actuator_ctrlrange[:5].T
        self.jp = np.zeros((3, self.model.nv))
        self.jr = np.zeros_like(self.jp)
        self.rotation_weight = .05 / np.sqrt(2)  # metre/radian at small angles

    def pose_jacobian(self, q):
        self.state.qpos[self.qadr[:5]] = q
        mujoco.mj_kinematics(self.model, self.state)
        mujoco.mj_comPos(self.model, self.state)
        mujoco.mj_jacSite(self.model, self.state, self.jp, self.jr, self.site)
        r = self.state.site_xmat[self.site].reshape(3, 3).copy()
        dr = np.cross(self.jr[:, self.dofs].T[:, None, :], r.T[None, :, :])
        return self.state.site_xpos[self.site].copy(), r, self.jp[:, self.dofs].copy(), dr.transpose(2, 1, 0).reshape(9, 5)

    def command(self, scene, target, gripper, target_rotation):
        current = scene['data']
        self.state.qpos[:] = current.qpos
        seed = np.clip(current.qpos[self.qadr[:5]], self.low, self.high)
        def residual(q):
            p, r, _, _ = self.pose_jacobian(q)
            return np.r_[p - target, self.rotation_weight * (r - target_rotation).ravel(), 1e-4 * (q - seed)]
        def jac(q):
            _, _, jp, jr = self.pose_jacobian(q)
            return np.vstack((jp, self.rotation_weight * jr, 1e-4 * np.eye(5)))
        result = least_squares(residual, seed, jac=jac, bounds=(self.low, self.high),
                               max_nfev=40, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        p, r, _, _ = self.pose_jacobian(result.x)
        scene.setdefault('ik_errors', []).append(float(np.linalg.norm(p - target)))
        scene.setdefault('ik_rotation_errors', []).append(float(angle_error(r, target_rotation)))
        scene.setdefault('ik_targets', []).append(target.copy())
        scene.setdefault('ik_rotation_targets', []).append(target_rotation.copy())
        compensation = current.qfrc_bias[self.dofs] / self.model.actuator_gainprm[:5, 0]
        return np.r_[result.x + compensation, gripper]
