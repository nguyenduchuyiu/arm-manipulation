"""E157 physical cuboid RGB-D renderer and rejection rules."""

import json
import numpy as np
from scipy.spatial.transform import Rotation
from unified_geometry.simulation.scene import (
    ASSETS,
    BASE,
    NAMES,
    visual_model,
    clip_planes,
)
import mujoco
from unified_geometry.geometry.projection import backproject

N, K = (4096, 128)


def overlap(ca, Ra, ha, cb, Rb, hb, margin=0.008):
    for axis in [*Ra.T, *Rb.T] + [np.cross(a, b) for a in Ra.T for b in Rb.T]:
        length = np.linalg.norm(axis)
        if length < 1e-09:
            continue
        axis = axis / length
        radius = (ha * abs(Ra.T @ axis)).sum() + (hb * abs(Rb.T @ axis)).sum() + margin
        if abs((cb - ca) @ axis) > radius:
            return False
    return True


def valid_dimensions(rows, scales):
    centers = np.array([r["position"] for r in rows])
    half = np.array([r["side_m"] for r in rows])[:, None] * scales * 0.5
    rotations = [Rotation.from_quat(np.roll(r["quaternion_wxyz"], -1)).as_matrix() for r in rows]
    if any((centers[i, 2] - (abs(rotations[i][2]) * half[i]).sum() < 0.004 for i in range(4))):
        return False
    return not any(
        (
            overlap(centers[i], rotations[i], half[i], centers[j], rotations[j], half[j])
            for i in range(4)
            for j in range(i)
        )
    )


class Renderer:
    def __init__(self):
        self.model = visual_model()
        clip_planes(self.model)
        m = self.model
        m.vis.global_.offwidth = 320
        m.vis.global_.offheight = 240
        self.state = mujoco.MjData(m)
        initial = json.loads((ASSETS / "initial_state.json").read_text())
        for key in ("qpos", "qvel", "ctrl"):
            getattr(self.state, key)[:] = initial[key]
        self.base_state = self.state.qpos.copy()
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        self.rgb = mujoco.Renderer(m, height=240, width=320)
        samples = m.vis.quality.offsamples
        m.vis.quality.offsamples = 0
        self.seg = mujoco.Renderer(m, height=240, width=320)
        self.seg.enable_segmentation_rendering()
        m.vis.quality.offsamples = samples
        bodies = [m.body(n).id for n in NAMES[:4]]
        self.addresses = [int(m.jnt_qposadr[m.body_jntadr[b]]) for b in bodies]
        self.geoms = [[m.geom(n + "_visual").id, m.geom(n + "_collision").id] for n in NAMES[:4]]
        self.original_geom_size = m.geom_size.copy()
        self.lut = np.zeros(m.ngeom, np.uint8)
        for i, b in enumerate(bodies, 1):
            self.lut[m.geom_bodyid == b] = i
        self.intrinsics = np.array(initial["intrinsics"])
        self.transform = np.array(initial["camera_to_base"])
        mujoco.mj_forward(m, self.state)
        cid = m.camera("cam_front").id
        actual = np.eye(4)
        actual[:3, :3] = self.state.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        actual[:3, 3] = self.state.cam_xpos[cid] - BASE
        np.testing.assert_allclose(actual, self.transform, atol=1e-12, rtol=0)

    def render(self, rows, scales, rng):
        m = self.model
        state = self.state
        state.qpos[:] = self.base_state
        for row, scale, address, gids in zip(rows, scales, self.addresses, self.geoms):
            state.qpos[address : address + 3] = row["position"]
            state.qpos[address + 3 : address + 7] = row["quaternion_wxyz"]
            m.geom_size[gids] = np.array(row["side_m"]) * scale / 2
        other = np.ones(m.ngeom, dtype=bool)
        other[np.array(self.geoms).flatten()] = False
        assert np.array_equal(m.geom_size[other], self.original_geom_size[other])
        mujoco.mj_forward(m, state)
        self.seg.update_scene(state, camera="cam_front", scene_option=self.option)
        seg = self.seg.render()
        good = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        labels = np.zeros((240, 320), np.uint8)
        labels[good] = self.lut[seg[..., 0][good]]
        visible = [int((labels == j + 1).sum()) for j in range(4)]
        if min(visible) < 16:
            return None
        self.rgb.update_scene(state, camera="cam_front", scene_option=self.option)
        rgb = self.rgb.render().copy()
        self.rgb.enable_depth_rendering()
        depth = self.rgb.render().copy()
        self.rgb.disable_depth_rendering()
        full, uv = backproject(depth, self.intrinsics, self.transform)
        chosen = rng.choice(len(full), N, replace=False)
        full_labels = labels[uv[:, 1], uv[:, 0]]
        targets = []
        for task in range(4):
            cloud = full[full_labels == task + 1]
            targets.append(cloud[rng.choice(len(cloud), K, replace=len(cloud) < K)])
        return (
            dict(
                rgb=rgb,
                depth_m=depth,
                scene_points=full[chosen],
                pixel_uv=uv[chosen],
                target_points=np.stack(targets),
                instance_labels=labels,
            ),
            visible,
        )

    def close(self):
        self.rgb.close()
        self.seg.close()
