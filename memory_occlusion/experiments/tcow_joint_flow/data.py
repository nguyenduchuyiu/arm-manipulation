"""25 Hz RGB-D / mask / action clips from scene-disjoint episodes."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


def split_rows(root):
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    train = [row for row in rows if row["group"] == "standard" and row["split"] == "train"]
    val = [row for row in rows if row["group"] == "standard" and row["split"] == "val"]
    test = [row for row in rows if row["split"] == "test"]
    if (len(train), len(val), len(test)) != (640, 80, 160):
        raise ValueError("expected 640 train, 80 validation, and 160 untouched test episodes")
    if len({row["seed"] for row in train}) != 160 or len({row["seed"] for row in val}) != 20:
        raise ValueError("scene split is incomplete")
    if {row["seed"] for row in train} & {row["seed"] for row in val + test}:
        raise ValueError("scene seed leaked from train")
    for row in train + val:
        if not (root / row["path"] / "episode.json").is_file():
            raise FileNotFoundError(root / row["path"] / "episode.json")
    return train, val


def policy_statistics(root, rows):
    """Mean/std on valid TRAIN action frames, without decoding RGB or depth."""
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
    result = {"valid_train_frames": total}
    for name in sums:
        mean = sums[name] / total
        std = np.sqrt(np.maximum(squares[name] / total - mean**2, 0))
        # Constant joints should stay in their original scale.
        std = np.where(std < 1e-3, 1.0, std)
        result[f"{name}_mean"] = mean.tolist()
        result[f"{name}_std"] = std.tolist()
    return result


def load_episode(path: Path):
    meta = json.loads((path / "episode.json").read_text())
    if meta["fps"] != 25 or meta["resolution"] != [240, 320]:
        raise ValueError(f"expected 25 Hz 240x320: {path}")
    rgb = np.stack(list(iio.imiter(path / "rgb.mp4", plugin="pyav")))
    with np.load(path / "observation.npz") as z:
        depth = z["depth_m"]
        proprio = z["joint_position"]
    with np.load(path / "supervision.npz") as z:
        query = z["mask"][0] == 2
        action = z["expert_action"]
        valid = z["action_valid"]
    with np.load(path / "tcow_labels.npz") as z:
        masks = z["mask"]
    n = len(rgb)
    if any(len(x) != n for x in (depth, proprio, action, valid, masks)) or not query.any():
        raise ValueError(f"episode arrays disagree: {path}")
    valid_ends = np.flatnonzero(np.convolve(valid.astype(np.int32),
                             np.ones(25, np.int32), mode="valid") == 25)
    valid_ends = valid_ends[valid_ends >= 29]
    if not len(valid_ends):
        raise ValueError(f"no 25-step valid action chunk: {path}")
    return dict(rgb=rgb, depth=depth, proprio=proprio, action=action,
                mask=masks, query=query, valid_ends=valid_ends, name=path.name,
                release=int(meta["decision_frames"]["t_obj"]) if "t_obj" in meta["decision_frames"] else 0)


def clip(episode, end: int, device: str):
    indices = np.rint(np.linspace(0, end, 30)).astype(np.int64)
    rgb = torch.from_numpy(episode["rgb"][indices].copy()).permute(3, 0, 1, 2).to(device)
    rgb = (rgb.float() / 255.0 - .45) / .225
    depth = torch.from_numpy(episode["depth"][indices].copy()).to(device)
    depth = ((depth.clamp(.4, 1.6) - 1.0) / .6)[None]
    rgbd = torch.cat((rgb, depth), dim=0)[None]
    query = torch.zeros((1, 1, 30, 240, 320), device=device)
    query[0, 0, 0] = torch.from_numpy(episode["query"]).to(device)
    masks = torch.from_numpy(episode["mask"][indices].copy()).permute(1, 0, 2, 3)[None].float().to(device)
    proprio = torch.from_numpy(episode["proprio"][end].copy())[None].to(device)
    action = torch.from_numpy(episode["action"][end:end + 25].copy())[None].to(device)
    return rgbd, query, masks, proprio, action


def load_policy_episode(path: Path):
    """Load 25 Hz policy data without the large three-mask supervision array."""
    meta = json.loads((path / "episode.json").read_text())
    if meta["fps"] != 25 or meta["resolution"] != [240, 320]:
        raise ValueError(f"expected 25 Hz 240x320: {path}")
    rgb = np.stack(list(iio.imiter(path / "rgb.mp4", plugin="pyav")))
    with np.load(path / "observation.npz") as z:
        depth = z["depth_m"]
        proprio = z["joint_position"]
    with np.load(path / "supervision.npz") as z:
        action = z["expert_action"]
        valid = z["action_valid"]
    query = np.asarray(Image.open(path / "query_mask.png")) > 0
    n = len(rgb)
    if any(len(x) != n for x in (depth, proprio, action, valid)) or not query.any():
        raise ValueError(f"episode arrays disagree: {path}")
    valid_ends = np.flatnonzero(np.convolve(valid.astype(np.int32),
                             np.ones(25, np.int32), mode="valid") == 25)
    decision = int(meta["decision_frames"]["t_occ"])
    valid_ends = valid_ends[valid_ends >= decision]
    events = (decision, int(meta["semantic_transition_frames"]["cover_released"]),
              int(meta["decision_frames"]["t_obj"]))
    if len(valid_ends) == 0 or not np.isin(events[0], valid_ends):
        raise ValueError(f"no valid 25-action chunk at policy start: {path}")
    return dict(rgb=rgb, depth=depth, proprio=proprio, action=action, action_valid=valid,
                query=query, valid_ends=valid_ends, events=events, name=path.name)


def policy_clip(episode, end: int, device: str):
    """Return one query-to-current TCOW clip and its 25 future action commands."""
    indices = np.rint(np.linspace(0, end, 30)).astype(np.int64)
    rgb = torch.from_numpy(episode["rgb"][indices].copy()).permute(3, 0, 1, 2).to(device)
    rgb = (rgb.float() / 255.0 - .45) / .225
    depth = torch.from_numpy(episode["depth"][indices].copy()).to(device)
    depth = ((depth.clamp(.4, 1.6) - 1.0) / .6)[None]
    rgbd = torch.cat((rgb, depth), dim=0)[None]
    query = torch.zeros((1, 1, 30, 240, 320), device=device)
    query[0, 0, 0] = torch.from_numpy(episode["query"]).to(device)
    proprio = torch.from_numpy(episode["proprio"][end].copy())[None].to(device)
    action = torch.from_numpy(episode["action"][end:end + 25].copy())[None].to(device)
    return rgbd, query, proprio, action


def load_joint_episode(path: Path):
    episode = load_policy_episode(path)
    with np.load(path / "tcow_labels.npz") as z:
        episode["mask"] = z["mask"]
    if len(episode["mask"]) != len(episode["rgb"]):
        raise ValueError(f"TCOW labels disagree with RGB: {path}")
    return episode


def joint_clip(episode, end: int, device: str):
    rgbd, query, proprio, action = policy_clip(episode, end, device)
    indices = np.rint(np.linspace(0, end, 30)).astype(np.int64)
    masks = torch.from_numpy(episode["mask"][indices].copy()).permute(1, 0, 2, 3)[None]
    action_frames = torch.from_numpy((indices >= episode["events"][0]).copy())[None]
    return rgbd, query, masks.float().to(device), proprio, action, action_frames.to(device)


def sample_plan(valid, decision: int, name: str):
    """Sample both phases at 10-frame intervals, with 25-step action targets."""
    if not 0 < decision < len(valid) or valid[:decision].any() or not valid[decision]:
        raise ValueError(f"unexpected action_valid boundary: {name}")
    action_stop = decision + int(np.flatnonzero(~valid[decision:])[0]) if (~valid[decision:]).any() else len(valid)
    if valid[action_stop:].any():
        raise ValueError(f"noncontiguous valid actions: {name}")
    context = list(range(0, decision, 10))
    if context[-1] != decision - 1:
        context.append(decision - 1)
    return ([(end, False) for end in context] +
            [(end, True) for end in range(decision, action_stop, 10)])


def sample_index(path: Path):
    meta = json.loads((path / "episode.json").read_text())
    with np.load(path / "supervision.npz") as z:
        valid = z["action_valid"]
    return sample_plan(valid, int(meta["decision_frames"]["t_occ"]), path.name)


def training_samples(episode):
    return sample_plan(episode["action_valid"], episode["events"][0], episode["name"])


def training_clip(episode, end: int, has_action: bool, device: str):
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
    return rgbd, query, masks, proprio, action, valid_steps
