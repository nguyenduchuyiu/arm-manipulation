"""Render one fixed, close-up reference image for each HOPE target."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from OATFlow.environment.env import SCENE_XML, TARGETS, TARGET_HALF_HEIGHT, configure_mvp_model


REFERENCE_SIZE = 128


def create_references(root: Path) -> None:
    directory = root / "references"
    directory.mkdir(parents=True, exist_ok=True)
    if all((directory / f"{name}.png").is_file() for name in TARGETS):
        return

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    configure_mvp_model(model)
    model.vis.global_.offheight = 512
    model.vis.global_.offwidth = 512
    data = mujoco.MjData(model)
    camera = model.camera("overview")
    camera.pos[:] = (0.5, -0.43, 0.13)
    camera.fovy[0] = 25.0
    forward = np.array((0.5, -0.23, 0.025)) - camera.pos
    forward /= np.linalg.norm(forward)
    right = np.array((1.0, 0.0, 0.0))
    up = np.cross(-forward, right)
    mujoco.mju_mat2Quat(camera.quat, np.column_stack((right, up, -forward)).flatten())
    renderer = mujoco.Renderer(model, height=512, width=512)

    for name in TARGETS:
        mujoco.mj_resetData(model, data)
        for other in TARGETS:
            qadr = model.jnt_qposadr[model.joint(f"{other}_joint").id]
            data.qpos[qadr:qadr + 3] = (
                (0.5, -0.23, TARGET_HALF_HEIGHT[other]) if other == name
                else (3.0, 3.0, 0.2)
            )
            data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        for cover in ("cover_a", "cover_b"):
            qadr = model.jnt_qposadr[model.joint(f"{cover}_joint").id]
            data.qpos[qadr:qadr + 3] = (4.0, 4.0, 0.2)
            data.qpos[qadr + 3:qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="overview")
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="overview")
        segmentation = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        mask = segmentation[:, :, 0] == model.geom(f"{name}_visual").id
        ys, xs = np.where(mask)
        if len(xs) == 0:
            raise RuntimeError(f"{name} is not visible in close-up reference")
        side = int(max(np.ptp(xs), np.ptp(ys)) * 1.2) + 1
        cx, cy = int(np.mean((xs.min(), xs.max()))), int(np.mean((ys.min(), ys.max())))
        x0, y0 = cx - side // 2, cy - side // 2
        canvas = np.full((side, side, 3), 128, dtype=np.uint8)
        src_x0, src_y0 = max(0, x0), max(0, y0)
        src_x1, src_y1 = min(512, x0 + side), min(512, y0 + side)
        patch = canvas[src_y0 - y0:src_y1 - y0, src_x0 - x0:src_x1 - x0]
        src_mask = mask[src_y0:src_y1, src_x0:src_x1]
        patch[src_mask] = rgb[src_y0:src_y1, src_x0:src_x1][src_mask]
        Image.fromarray(canvas).resize((REFERENCE_SIZE, REFERENCE_SIZE), Image.Resampling.LANCZOS).save(
            directory / f"{name}.png"
        )
    renderer.close()
