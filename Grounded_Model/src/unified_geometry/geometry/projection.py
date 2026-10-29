"""Depth backprojection in robot base coordinates."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np


def backproject(depth, intrinsics, transform):
    v, u = np.nonzero(np.isfinite(depth) & (depth > 0) & (depth <= 6.5535))
    d = depth[v, u].astype(np.float64)
    camera = np.c_[
        (u - intrinsics[0, 2]) * d / intrinsics[0, 0],
        (v - intrinsics[1, 2]) * d / intrinsics[1, 1],
        d,
    ]
    points = (camera @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)
    return (points, np.c_[u, v].astype(np.uint16))
