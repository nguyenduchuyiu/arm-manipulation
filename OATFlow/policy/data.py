"""25 Hz RGB, TCOW masks and absolute actions from scene-disjoint episodes."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import sys

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from OATFlow.dataset.actions import chunk_starts


def split_rows(root):
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    train = [row for row in rows if row["group"] == "standard" and row["split"] == "train"]
    val = [row for row in rows if row["group"] == "standard" and row["split"] == "val"]
    if not train:
        raise ValueError("no standard/train context episodes")
    partitions = {split: {r["seed"] for r in rows if r["split"] == split} for split in ("train", "val", "test")}
    if any(partitions[a] & partitions[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("scene seed leaked between splits")
    for row in train + val:
        if not (root / row["path"] / "episode.json").is_file():
            raise FileNotFoundError(root / row["path"] / "episode.json")
    return train, val


def require_wrist_data(root, rows):
    for row in rows:
        path = root / row["path"] / "wrist_rgb.mp4"
        if not path.is_file():
            raise FileNotFoundError(f"missing wrist video; collect synchronized multiview data: {path}")


def read_rgb(path, count, resolution, frames=None):
    """Decode directly into one allocation; optionally retain only sampled frames."""
    selected = np.arange(count) if frames is None else frames
    result = np.empty((len(selected), *resolution, 3), dtype=np.uint8)
    kept = decoded = 0
    for index, frame in enumerate(iio.imiter(path, plugin="pyav", thread_count=2, thread_type="FRAME")):
        if index >= count or frame.shape != (*resolution, 3) or frame.dtype != np.uint8:
            raise ValueError(f"video frames disagree with metadata: {path}")
        if kept < len(selected) and index == selected[kept]:
            result[kept] = frame
            kept += 1
        decoded += 1
    if decoded != count or kept != len(selected):
        raise ValueError(f"video frames disagree with metadata: {path}")
    return result


def policy_statistics(root, rows):
    """Mean/std on valid TRAIN action frames, without decoding RGB."""
    if (root / "normalization.json").is_file():
        statistics = json.loads((root / "normalization.json").read_text())
        if statistics["action_representation"] != "absolute_joint" or any(r["split"] != "train" or r["group"] != "standard" for r in rows):
            raise ValueError("action normalization must use standard train data")
        return statistics
    total = 0
    sums = {name: np.zeros(6, np.float64) for name in ("state", "action")}
    squares = {name: np.zeros(6, np.float64) for name in sums}
    for row in tqdm(rows, desc="policy normalization", unit="episode", mininterval=5, file=sys.stdout):
        path = root / row["path"]
        with np.load(path / "supervision.npz") as z:
            valid = z["action_valid"]
            action = z["expert_action"][valid].astype(np.float64)
        with np.load(path / "observation.npz") as z:
            state = z["joint_position"][valid].astype(np.float64)
        total += len(action)
        for name, value in (("state", state), ("action", action)):
            sums[name] += value.sum(axis=0)
            squares[name] += np.square(value).sum(axis=0)
    if total == 0:
        raise ValueError("no train action frames for normalization")
    result = {"valid_train_frames": total, "action_representation": "absolute_joint"}
    for name in sums:
        mean = sums[name] / total
        std = np.sqrt(np.maximum(squares[name] / total - mean**2, 0))
        # Constant joints should stay in their original scale.
        std = np.where(std < 1e-3, 1.0, std)
        result[f"{name}_mean"] = mean.tolist()
        result[f"{name}_std"] = std.tolist()
    return result


def visual_clip(episode, indices, device):
    rgb_indices = indices
    if "rgb_frames" in episode:
        rgb_indices = np.searchsorted(episode["rgb_frames"], indices)
        if not np.array_equal(episode["rgb_frames"][rgb_indices], indices):
            raise ValueError("missing overview frames in prepared batch")
    rgb = torch.from_numpy(episode["rgb"][rgb_indices, 40:280].copy()).permute(3, 0, 1, 2).to(device)
    rgb = (rgb.float() / 255.0 - .45) / .225
    return rgb[None]


def load_policy_episode(path: Path, require_wrist=False, training_cache=False,
                        videos=None):
    """Load 25 Hz policy data without the large three-mask supervision array."""
    meta = json.loads((path / "episode.json").read_text())
    if meta["fps"] != 25 or meta["resolution"] != [320, 320] or meta.get("tcow_resolution") != [240, 320]:
        raise ValueError(f"expected full RGB with 240x320 TCOW labels: {path}")
    if videos is None:
        rgb = read_rgb(path / "rgb.mp4", meta["frames"], (320, 320))
    else:
        from OATFlow.policy.loader import LeRobotVideoFrames
        rgb = LeRobotVideoFrames(videos[(path / "rgb.mp4").resolve()], meta["frames"], (320, 320))
    with np.load(path / "observation.npz") as z:
        proprio = z["joint_position"]
    with np.load(path / "supervision.npz") as z:
        action = z["expert_action"]
        valid = z["action_valid"]
        phase = z["phase"]
    query = np.asarray(Image.open(path / "query_mask.png")) > 0
    n = len(rgb)
    if any(len(x) != n for x in (proprio, action, valid)) or not query.any():
        raise ValueError(f"episode arrays disagree: {path}")
    valid_ends = np.flatnonzero(np.convolve(valid.astype(np.int32),
                             np.ones(25, np.int32), mode="valid") == 25)
    decision = int(meta["decision_frames"]["t_occ"])
    valid_ends = valid_ends[valid_ends >= decision]
    events = (decision, int(meta["semantic_transition_frames"]["cover_released"]),
              int(meta["decision_frames"]["t_obj"]))
    if len(valid_ends) == 0 or not np.isin(events[0], valid_ends):
        raise ValueError(f"no valid 25-action chunk at policy start: {path}")
    episode = dict(rgb=rgb, proprio=proprio, action=action, action_valid=valid, phase=phase,
                   query=query, valid_ends=valid_ends, events=events, name=path.name)
    if require_wrist:
        wrist_meta = meta.get("wrist_camera")
        if wrist_meta is None or wrist_meta["fps"] != 25 or wrist_meta["resolution"] != [320, 320]:
            raise ValueError(f"missing synchronized 25 Hz wrist camera metadata: {path}")
        if wrist_meta["frames"] != n:
            raise ValueError(f"wrist frames disagree with overview: {path}")
        if videos is not None:
            episode["wrist_rgb"] = LeRobotVideoFrames(videos[(path / "wrist_rgb.mp4").resolve()], n, (320, 320))
        else:
            frames = np.array([end for end, _ in training_samples(episode)]) if training_cache else None
            episode["wrist_rgb"] = read_rgb(path / "wrist_rgb.mp4", n, (320, 320), frames)
            if training_cache:
                episode["wrist_frames"] = frames
    return episode


def policy_clip(episode, end: int, device: str):
    """Return one query-to-current TCOW clip and its 25 future action commands."""
    indices = np.rint(np.linspace(0, end, 30)).astype(np.int64)
    rgbd = visual_clip(episode, indices, device)
    query = torch.zeros((1, 1, 30, 240, 320), device=device)
    query[0, 0, 0] = torch.from_numpy(episode["query"].copy()).to(device)
    proprio = torch.from_numpy(episode["proprio"][end].copy())[None].to(device)
    action = torch.from_numpy(episode["action"][end:end + 25].copy())[None].to(device)
    return rgbd, query, proprio, action


@lru_cache(maxsize=8)
def packed_tcow_masks(path):
    """Reuse compact masks across microbatches from the same episode cluster."""
    with np.load(path / "tcow_labels.npz") as z:
        masks = z["mask"]
    if masks.dtype != np.bool_:
        raise ValueError(f"expected binary TCOW masks: {path}")
    packed = np.packbits(masks, axis=-1)
    packed.setflags(write=False)
    return packed


def load_joint_episode(path: Path, require_wrist=False, training_cache=False,
                       videos=None):
    episode = load_policy_episode(path, require_wrist=require_wrist, training_cache=training_cache,
                                  videos=videos)
    if training_cache:
        episode["mask_packed"] = packed_tcow_masks(path)
        count = len(episode["mask_packed"])
    else:
        with np.load(path / "tcow_labels.npz") as z:
            masks = z["mask"]
        episode["mask"] = masks
        count = len(masks)
    if count != len(episode["rgb"]):
        raise ValueError(f"TCOW labels disagree with RGB: {path}")
    return episode


def joint_clip(episode, end: int, device: str):
    rgbd, query, proprio, action = policy_clip(episode, end, device)
    indices = np.rint(np.linspace(0, end, 30)).astype(np.int64)
    if "mask_packed" in episode:
        masks = torch.from_numpy(np.unpackbits(episode["mask_packed"][indices], axis=-1, count=320))
        masks = masks.permute(1, 0, 2, 3)[None]
    else:
        masks = (torch.from_numpy(episode["mask"][indices].copy()).permute(1, 0, 2, 3)[None]
                 if "mask" in episode else torch.zeros((1, 3, 30, 1, 1)))
    action_frames = torch.from_numpy((indices >= episode["events"][0]).copy())[None]
    return rgbd, query, masks.float().to(device), proprio, action, action_frames.to(device)


def sample_plan(valid, decision: int, name: str):
    """Every context and expert frame; only expert frames carry action loss."""
    starts, _ = chunk_starts(valid, decision, name)
    context = range(decision)
    return ([(end, False) for end in context] +
            [(int(end), True) for end in starts])


def tracking_starts(frames):
    """Train TCOW masks at every context and expert frame."""
    if frames < 1:
        raise ValueError("positive frame count required")
    return list(range(frames))


def sample_index(path: Path):
    meta = json.loads((path / "episode.json").read_text())
    with np.load(path / "supervision.npz") as z:
        valid = z["action_valid"]
    return sample_plan(valid, int(meta["decision_frames"]["t_occ"]), path.name)


def training_samples(episode):
    return sample_plan(episode["action_valid"], episode["events"][0], episode["name"])


def training_clip(episode, end: int, has_action: bool, device: str, include_wrist=False):
    rgbd, query, masks, proprio, action, _frames = joint_clip(episode, end, device)
    valid_steps = torch.zeros((1, 25), dtype=torch.bool, device=device)
    if has_action:
        remaining = min(25, len(episode["action_valid"]) - end)
        valid_steps[0, :remaining] = torch.from_numpy(
            episode["action_valid"][end:end + remaining].copy()).to(device)
        if not valid_steps.any():
            raise ValueError(f"empty action chunk: {episode['name']} frame {end}")
    if action.shape[1] < 25:
        action = torch.cat((action, action[:, -1:].expand(-1, 25 - action.shape[1], -1)), dim=1)
    if has_action and not valid_steps.all():
        last = action[:, int(valid_steps.sum()) - 1:int(valid_steps.sum())]
        action = torch.where(valid_steps[:, :, None], action, last)
    result = (rgbd, query, masks, proprio, action, valid_steps)
    if include_wrist:
        from OATFlow.policy.model import preprocess_wrist_image
        index = end
        if "wrist_frames" in episode:
            index = int(np.searchsorted(episode["wrist_frames"], end))
            if index >= len(episode["wrist_frames"]) or episode["wrist_frames"][index] != end:
                raise ValueError(f"missing wrist frame: {episode['name']}, {end}")
        result += (preprocess_wrist_image(episode["wrist_rgb"][index], device),)
    return result
