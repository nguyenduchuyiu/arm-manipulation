"""RGB and the three predicted TCOW masks for closed-loop videos."""
import numpy as np
from PIL import Image, ImageDraw


def mask_panel(rgb, masks, frame_index, source_frame, query_mask=None):
    colors = ((255, 0, 200), (255, 160, 0), (0, 215, 255))
    labels = ("TCOW target", "TCOW occluder", "TCOW container")
    panels = []
    for channel in range(4):
        panel = rgb.copy()
        mask = query_mask if channel == 0 else masks[channel - 1]
        if mask is not None:
            color = (255, 230, 0) if channel == 0 else colors[channel - 1]
            panel[mask] = np.rint(panel[mask] * .4 + np.asarray(color) * .6).astype(np.uint8)
        panels.append(panel)
    display = Image.fromarray(np.concatenate(panels, axis=1))
    draw = ImageDraw.Draw(display)
    width = rgb.shape[1]
    for index, label in enumerate(("RGB + GT query" if query_mask is not None else "RGB", *labels)):
        left = index * width
        draw.rectangle((left, 0, left + width - 1, 17), fill=(0, 0, 0))
        if index == 0:
            pixels = int(query_mask.sum()) if query_mask is not None else None
            text_color = (255, 230, 0) if query_mask is not None else (255, 255, 255)
        else:
            pixels = int(masks[index - 1].sum())
            text_color = colors[index - 1]
        title = f"{label}  {pixels} px" if pixels is not None else label
        draw.text((left + 4, 3), title, fill=text_color)
    draw.rectangle((0, rgb.shape[0] - 17, width - 1, rgb.shape[0] - 1), fill=(0, 0, 0))
    draw.text((4, rgb.shape[0] - 15), f"frame {frame_index} | TCOW source {source_frame}",
              fill=(255, 255, 255))
    return np.asarray(display)
