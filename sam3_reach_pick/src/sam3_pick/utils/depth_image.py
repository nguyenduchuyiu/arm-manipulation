"""Depth visualization with a fixed metric range."""

from matplotlib import colormaps
import numpy as np
from PIL import Image
from sam3_pick.data.rendering import UNIT

RANGE = (0.25, 1.25)
LUT = (colormaps["turbo_r"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def depth_image(codes):
    metres = codes.astype(np.float32) * UNIT
    bins = np.rint(np.clip((metres - RANGE[0]) / (RANGE[1] - RANGE[0]), 0, 1) * 255).astype(int)
    pixels = LUT[bins].copy()
    pixels[codes == 0] = (11, 20, 36)
    return Image.fromarray(pixels).resize((640, 480), Image.Resampling.NEAREST)
