"""Paired primitive substitutions inside original cuboid bounds, same observation rays."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import mujoco
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation
from unified_geometry.geometry.projection import backproject
import json
from unified_geometry.simulation.scene import ASSETS, visual_model

NAMES = ("RedCube", "BlueCube", "GreenCube", "YellowCube", "cover_a", "cover_b")
BASE = np.array([0.539369022999, 0.0639727628473, 0.0])
FAMILIES = ["sphere", "ellipsoid", "cylinder", "capsule"]


def specification(family, h):
    if family == "cuboid":
        return (int(mujoco.mjtGeom.mjGEOM_BOX), np.array(h))
    if family == "sphere":
        return (int(mujoco.mjtGeom.mjGEOM_SPHERE), np.array([min(h), 0.0, 0.0]))
    if family == "ellipsoid":
        return (int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), np.array(h))
    if family == "cylinder":
        return (int(mujoco.mjtGeom.mjGEOM_CYLINDER), np.array([min(h[:2]), h[2], 0.0]))
    assert family == "capsule"
    r = min(h[0], h[1], h[2] / 2)
    return (int(mujoco.mjtGeom.mjGEOM_CAPSULE), np.array([r, h[2] - r, 0.0]))


def analytic_residual(local, family, size):
    if family == "cuboid":
        return np.abs((np.abs(local) / size).max(1) - 1) * min(size)
    if family == "ellipsoid":
        return np.abs(np.linalg.norm(local / size, axis=1) - 1) * min(size)
    if family == "sphere":
        return np.abs(np.linalg.norm(local, axis=1) - size[0])
    if family == "capsule":
        near = local.copy()
        near[:, 2] -= np.clip(near[:, 2], -size[1], size[1])
        return np.abs(np.linalg.norm(near, axis=1) - size[0])
    assert family == "cylinder"
    return np.minimum(
        np.abs(np.linalg.norm(local[:, :2], axis=1) - size[0]),
        np.abs(np.abs(local[:, 2]) - size[1]),
    )


class Renderer:
    def __init__(self, K, T):
        self.model = visual_model()
        m = self.model
        m.vis.map.znear = 0.001 / m.stat.extent
        m.vis.map.zfar = 10.0 / m.stat.extent
        m.vis.global_.offwidth = 320
        m.vis.global_.offheight = 240
        self.state = mujoco.MjData(m)
        initial = json.loads((ASSETS / "initial_state.json").read_text())
        for key in ["qpos", "qvel", "ctrl"]:
            getattr(self.state, key)[:] = initial[key]
        self.original_state = self.state.qpos.copy()
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.rgb = mujoco.Renderer(m, height=240, width=320)
        samples = m.vis.quality.offsamples
        m.vis.quality.offsamples = 0
        self.seg = mujoco.Renderer(m, height=240, width=320)
        self.seg.enable_segmentation_rendering()
        m.vis.quality.offsamples = samples
        self.geoms = [
            [m.geom(name + "_visual").id, m.geom(name + "_collision").id] for name in NAMES[:4]
        ]
        self.addresses = [int(m.jnt_qposadr[m.body_jntadr[m.body(name).id]]) for name in NAMES[:4]]
        self.original_sizes = m.geom_size.copy()
        self.original_types = m.geom_type.copy()
        self.other = np.ones(m.ngeom, bool)
        self.other[np.array(self.geoms).ravel()] = False
        self.lut = np.zeros(m.ngeom, np.uint8)
        for i, name in enumerate(NAMES[:4], 1):
            self.lut[m.geom_bodyid == m.body(name).id] = i
        self.K = K
        self.T = T
        mujoco.mj_forward(m, self.state)
        camera = m.camera("cam_front").id
        actual = np.eye(4)
        actual[:3, :3] = self.state.cam_xmat[camera].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        actual[:3, 3] = self.state.cam_xpos[camera] - BASE
        np.testing.assert_allclose(actual, T, atol=1e-12, rtol=0)

    def render(self, meta, pixel_uv, family, seed):
        m = self.model
        s = self.state
        s.qpos[:] = self.original_state
        sizes = []
        for row, dimensions, address, geoms in zip(
            meta["cubes"], meta["dimensions_m"], self.addresses, self.geoms
        ):
            s.qpos[address : address + 3] = row["position"]
            s.qpos[address + 3 : address + 7] = row["quaternion_wxyz"]
            kind, size = specification(family, np.array(dimensions) / 2)
            sizes.append(size)
            m.geom_type[geoms] = kind
            m.geom_size[geoms] = size
        assert np.array_equal(m.geom_size[self.other], self.original_sizes[self.other])
        assert np.array_equal(m.geom_type[self.other], self.original_types[self.other])
        mujoco.mj_forward(m, s)
        self.seg.update_scene(s, camera="cam_front", scene_option=self.option)
        seg = self.seg.render()
        valid = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        labels = np.zeros((240, 320), np.uint8)
        labels[valid] = self.lut[seg[..., 0][valid]]
        self.rgb.update_scene(s, camera="cam_front", scene_option=self.option)
        rgb = self.rgb.render().copy()
        self.rgb.enable_depth_rendering()
        depth = self.rgb.render().copy()
        self.rgb.disable_depth_rendering()
        full, uv = backproject(depth, self.K, self.T)
        full_labels = labels[uv[:, 1], uv[:, 0]]
        uv = pixel_uv
        d = depth[uv[:, 1], uv[:, 0]].astype(float)
        assert np.isfinite(d).all() and (d > 0).all() and (d <= 6.5535).all()
        cam = np.c_[
            (uv[:, 0] - self.K[0, 2]) * d / self.K[0, 0],
            (uv[:, 1] - self.K[1, 2]) * d / self.K[1, 1],
            d,
        ]
        points = (cam @ self.T[:3, :3].T + self.T[:3, 3]).astype(np.float32)
        rng = np.random.default_rng(seed)
        target = []
        checks = []
        for task in range(4):
            surface = full[full_labels == task + 1]
            assert len(surface) > 0, (family, task)
            target.append(surface[rng.choice(len(surface), 128, replace=len(surface) < 128)])
            inner = binary_erosion(labels == task + 1, iterations=1)
            interior = full[inner[np.nonzero(np.isfinite(depth) & (depth > 0) & (depth <= 6.5535))]]
            if len(interior):
                row = meta["cubes"][task]
                R = Rotation.from_quat(np.roll(row["quaternion_wxyz"], -1)).as_matrix()
                local = (interior.astype(float) + BASE - np.array(row["position"])) @ R
                residual = analytic_residual(local, family, sizes[task]) * 1000
                pixel_depth = depth[
                    np.nonzero(np.isfinite(depth) & (depth > 0) & (depth <= 6.5535))
                ][inner[np.nonzero(np.isfinite(depth) & (depth > 0) & (depth <= 6.5535))]]
                footprint_mm = pixel_depth / min(self.K[0, 0], self.K[1, 1]) * 1000
                relative = residual / footprint_mm
                checks.append(
                    dict(
                        task=task,
                        interior_pixels=len(interior),
                        p95_residual_mm=float(np.quantile(residual, 0.95)),
                        max_residual_mm=float(residual.max()),
                        p95_residual_pixel_footprints=float(np.quantile(relative, 0.95)),
                    )
                )
                assert np.quantile(relative, 0.95) < 1.0, (family, task, checks[-1])
        info = dict(
            family=family,
            geom_sizes=np.array(sizes).tolist(),
            visible_pixels=[int((labels == t + 1).sum()) for t in range(4)],
            input_points_per_object=[
                int((labels[uv[:, 1], uv[:, 0]] == t + 1).sum()) for t in range(4)
            ],
            analytic_surface_checks=checks,
        )
        return (
            dict(
                rgb=rgb,
                depth_m=depth,
                instance_labels=labels,
                scene_points=points,
                pixel_uv=pixel_uv,
                target_points=np.stack(target),
            ),
            info,
        )

    def close(self):
        self.rgb.close()
        self.seg.close()
