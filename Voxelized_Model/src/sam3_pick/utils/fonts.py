"""Fonts used in recorded rollout videos."""

from PIL import ImageFont


def font(size, bold=False):
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf",
        size,
    )
