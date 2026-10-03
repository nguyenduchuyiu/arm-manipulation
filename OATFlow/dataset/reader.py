"""Load a decision observation without exposing future expert frames or labels."""
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def decision_observation(directory: Path, decision="t_occ", context_fps=8):
    inputs = json.loads((directory / "input.json").read_text())
    frame = inputs["decision_frames"][decision]
    with np.load(directory / inputs["observation"]) as data:
        times = data["timestamp_s"][:frame + 1]
        wanted = np.arange(0, times[-1], 1 / context_fps)
        right = np.clip(np.searchsorted(times, wanted), 1, len(times) - 1)
        left = right - 1
        indices = np.unique(np.where(wanted - times[left] <= times[right] - wanted, left, right))
        indices = indices[indices < frame]
        requested = set(indices.tolist()) | {frame}
        images = {}
        for i, rgb in enumerate(iio.imiter(directory / inputs["rgb_video"], plugin="pyav")):
            if i in requested:
                images[i] = rgb
            if i == frame:
                break
        depth = data["depth_m"]
        return {
            "context_rgb": np.stack([images[i] for i in indices]),
            "context_depth_m": depth[indices],
            "context_timestamp_s": times[indices],
            "joint_position": data["joint_position"][frame].copy(),
            "query_rgb": iio.imread(directory / inputs["query_rgb"]),
            "rgb": images[frame], "depth_m": depth[frame].copy(),
        }
