"""Change physical camera extrinsics; preserve exact paired cuboid scenes."""

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

BASE = np.array([0.539369022999, 0.0639727628473, 0.0])
NAMES = ("RedCube", "BlueCube", "GreenCube", "YellowCube", "cover_a", "cover_b")
PIVOT = np.array([0.54, -0.25, 0.08])
VIEWS = {
    "original": ("yaw", 0.0),
    "yaw_m15": ("yaw", -15.0),
    "yaw_p15": ("yaw", 15.0),
    "yaw_m30": ("yaw", -30.0),
    "yaw_p30": ("yaw", 30.0),
    "up15": ("elevation", 15.0),
}


class Renderer:
    def __init__(self, K, T):
        self.K = K
        self.original_T = T
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
        self.original_qpos = self.state.qpos.copy()
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.rgb = mujoco.Renderer(m, height=240, width=320)
        samples = m.vis.quality.offsamples
        m.vis.quality.offsamples = 0
        self.seg = mujoco.Renderer(m, height=240, width=320)
        self.seg.enable_segmentation_rendering()
        m.vis.quality.offsamples = samples
        self.camera = m.camera("cam_front").id
        assert int(m.cam_bodyid[self.camera]) == 0
        self.original_pos = m.cam_pos[self.camera].copy()
        self.original_quat = m.cam_quat[self.camera].copy()
        self.addresses = [int(m.jnt_qposadr[m.body_jntadr[m.body(name).id]]) for name in NAMES[:4]]
        self.geoms = [
            [m.geom(name + "_visual").id, m.geom(name + "_collision").id] for name in NAMES[:4]
        ]
        self.other = np.ones(m.ngeom, bool)
        self.other[np.array(self.geoms).ravel()] = False
        self.original_sizes = m.geom_size.copy()
        self.lut = np.zeros(m.ngeom, np.uint8)
        for task, name in enumerate(NAMES[:4], 1):
            self.lut[m.geom_bodyid == m.body(name).id] = task
        mujoco.mj_forward(m, self.state)
        np.testing.assert_allclose(self.transform(), T, atol=1e-12, rtol=0)

    def transform(self):
        T = np.eye(4)
        T[:3, :3] = self.state.cam_xmat[self.camera].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = self.state.cam_xpos[self.camera] - BASE
        return T

    def view(self, name):
        kind, degrees = VIEWS[name]
        axis = np.array([0.0, 0.0, 1.0]) if kind == "yaw" else -self.original_T[:3, 0]
        S = Rotation.from_rotvec(axis * np.deg2rad(degrees)).as_matrix()
        self.model.cam_pos[self.camera] = PIVOT + S @ (self.original_pos - PIVOT)
        if degrees == 0:
            self.model.cam_quat[self.camera] = self.original_quat
        else:
            self.model.cam_quat[self.camera] = np.roll(
                Rotation.from_matrix(
                    S @ self.original_T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
                ).as_quat(),
                1,
            )
        mujoco.mj_forward(self.model, self.state)
        T = self.transform()
        expected = self.original_T.copy()
        expected[:3, :3] = S @ self.original_T[:3, :3]
        expected[:3, 3] = PIVOT + S @ (self.original_pos - PIVOT) - BASE
        np.testing.assert_allclose(T, expected, atol=1e-12, rtol=0)
        assert abs(np.linalg.det(T[:3, :3]) - 1) < 1e-12
        point = (PIVOT - BASE - T[:3, 3]) @ T[:3, :3]
        uv = (point @ self.K.T)[:2] / point[2]
        ref = (PIVOT - BASE - self.original_T[:3, 3]) @ self.original_T[:3, :3]
        reference = (ref @ self.K.T)[:2] / ref[2]
        assert np.linalg.norm(uv - reference) < 1e-09
        return T

    def render(self, meta, view, scene):
        m = self.model
        s = self.state
        s.qpos[:] = self.original_qpos
        for row, dimensions, address, geoms in zip(
            meta["cubes"], meta["dimensions_m"], self.addresses, self.geoms
        ):
            s.qpos[address : address + 3] = row["position"]
            s.qpos[address + 3 : address + 7] = row["quaternion_wxyz"]
            m.geom_size[geoms] = np.array(dimensions) / 2
        assert np.array_equal(m.geom_size[self.other], self.original_sizes[self.other])
        T = self.view(view)
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
        full, full_uv = backproject(depth, self.K, T)
        assert len(full) >= 4096
        chosen = np.random.default_rng(17564000 + scene).choice(len(full), 4096, replace=False)
        points = full[chosen]
        pixel_uv = full_uv[chosen]
        full_labels = labels[full_uv[:, 1], full_uv[:, 0]]
        gt_rng = np.random.default_rng(17664000 + scene)
        targets = []
        visible = []
        checks = []
        for task in range(4):
            surface = full[full_labels == task + 1]
            visible.append(len(surface))
            targets.append(
                surface[gt_rng.choice(len(surface), 128, replace=len(surface) < 128)]
                if len(surface)
                else np.full((128, 3), np.nan, np.float32)
            )
            inner = binary_erosion(labels == task + 1, iterations=1)
            which = inner[full_uv[:, 1], full_uv[:, 0]]
            if which.any():
                row = meta["cubes"][task]
                R = Rotation.from_quat(np.roll(row["quaternion_wxyz"], -1)).as_matrix()
                h = np.array(meta["dimensions_m"][task]) / 2
                local = (full[which].astype(float) + BASE - np.array(row["position"])) @ R
                residual = np.abs((np.abs(local) / h).max(1) - 1) * h.min() * 1000
                footprint = (
                    depth[full_uv[which, 1], full_uv[which, 0]]
                    / min(self.K[0, 0], self.K[1, 1])
                    * 1000
                )
                checks.append(
                    dict(
                        task=task,
                        interior_pixels=int(which.sum()),
                        p95_residual_mm=float(np.quantile(residual, 0.95)),
                        p95_pixel_footprints=float(np.quantile(residual / footprint, 0.95)),
                    )
                )
                assert checks[-1]["p95_pixel_footprints"] < 1, (view, scene, checks[-1])
        info = dict(
            view=view,
            scene=scene,
            camera_to_base=T.tolist(),
            visible_pixels=visible,
            input_points_per_object=[
                int((full_labels[chosen] == task + 1).sum()) for task in range(4)
            ],
            analytic_checks=checks,
        )
        return (
            dict(
                rgb=rgb,
                depth_m=depth,
                scene_points=points,
                pixel_uv=pixel_uv,
                instance_labels=labels,
                target_points=np.stack(targets),
                target_valid=np.array(visible) > 0,
            ),
            info,
        )

    def close(self):
        self.rgb.close()
        self.seg.close()
