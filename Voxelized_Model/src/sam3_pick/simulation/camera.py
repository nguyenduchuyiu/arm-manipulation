"""Calibrated dense front-depth geometry, shared by training and online rollout."""

import numpy as np

BASE = np.array([0.539369022999, 0.0639727628473, 0.0])


def camera_calibration(model, state):
    cid = model.camera("cam_front").id
    focal = 120 / np.tan(np.deg2rad(model.cam_fovy[cid]) / 2)
    K = np.array([[focal, 0, 159.5], [0, focal, 119.5], [0, 0, 1.0]], np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = state.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = state.cam_xpos[cid] - BASE
    return (K, transform)
