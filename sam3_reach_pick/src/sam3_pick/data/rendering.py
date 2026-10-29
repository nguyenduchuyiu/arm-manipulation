from pathlib import Path
import subprocess
import mujoco
import numpy as np
from sam3_pick.simulation.scene import NAMES
from sam3_pick.simulation.render import visual_model
from sam3_pick.simulation.wrist_camera import Fisheye

VIEWS = ("front", "wrist")
DEPTH_VIEWS = ("front",)
SCHEMA = "front_rgbd_wrist_fisheye_rgb_v1"
CAMERAS = ("cam_front", "wrist")
UNIT = 0.002
MAX_DEPTH = 6.5535
CLIP_NEAR = 0.001
CLIP_FAR = 10.0


def clip_planes(model):
    model.vis.map.znear = CLIP_NEAR / model.stat.extent
    model.vis.map.zfar = CLIP_FAR / model.stat.extent


def initialize(root):
    global ROOT, MODEL, STATE, RGB, SEG, OPTION, ONLY, GEOMS, LUT, WRIST
    ROOT = Path(root)
    MODEL = visual_model(ROOT / "scene.xml", object_bodies=NAMES)
    clip_planes(MODEL)
    MODEL.vis.global_.offwidth = 320
    MODEL.vis.global_.offheight = 256
    STATE = mujoco.MjData(MODEL)
    RGB = mujoco.Renderer(MODEL, height=240, width=320)
    samples = MODEL.vis.quality.offsamples
    MODEL.vis.quality.offsamples = 0
    SEG = mujoco.Renderer(MODEL, height=240, width=320)
    MODEL.vis.quality.offsamples = samples
    SEG.enable_segmentation_rendering()
    WRIST = Fisheye(MODEL)
    OPTION = mujoco.MjvOption()
    OPTION.geomgroup[3] = 0
    ONLY = mujoco.MjvOption()
    ONLY.geomgroup[:] = 0
    ONLY.geomgroup[5] = 1
    LUT = np.zeros(MODEL.ngeom, np.uint8)
    GEOMS = {}
    for index, name in enumerate(NAMES, 1):
        body = MODEL.body(name).id
        GEOMS[index] = np.flatnonzero((MODEL.geom_bodyid == body) & (MODEL.geom_rgba[:, 3] > 0))
        LUT[MODEL.geom_bodyid == body] = index
    for g in range(MODEL.ngeom):
        if MODEL.geom_bodyid[g] != 0 and MODEL.body(int(MODEL.geom_bodyid[g])).name not in NAMES:
            LUT[g] = 7
    LUT[MODEL.geom("table_top").id] = 8


def depth_encoder(path):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",
            "-s",
            "320x240",
            "-r",
            "25",
            "-i",
            "-",
            "-an",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-coder",
            "1",
            "-context",
            "1",
            "-g",
            "1",
            "-slices",
            "4",
            "-slicecrc",
            "1",
            "-threads",
            "1",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
