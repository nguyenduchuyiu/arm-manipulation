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

from OATFlow.dataset.actions import chunk_starts


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


def expand_demonstrations(root, rows):
    """Visit every attached recovery demo once per epoch, within its parent split."""
    expanded = []
    seen = set()
    for row in rows:
        if row["split"] != "train" or row["group"] != "standard":
            raise ValueError("demonstration expansion is only for standard/train")
        parent = root / row["path"]
        meta = json.loads((parent / "episode.json").read_text())
        if not meta["success"] or any(meta[key] != row[column] for key, column in
                (("seed", "seed"), ("split", "split"), ("layout", "group"))):
            raise ValueError(f"parent episode differs from manifest scene/split: {parent}")
        candidates = [("base", parent)]
        manifest = parent / "demonstrations.json"
        if manifest.is_file():
            contents = json.loads(manifest.read_text())
            if contents["version"] != 1:
                raise ValueError(f"unsupported demonstrations manifest: {manifest}")
            ids = {"base"}
            for demo in contents["demonstrations"]:
                path = (parent / demo["path"]).resolve()
                if demo["id"] in ids or not path.is_relative_to((parent / "demos").resolve()):
                    raise ValueError(f"invalid demonstration ID/path: {manifest}")
                ids.add(demo["id"])
                extra = json.loads((path / "episode.json").read_text())
                if not extra["success"] or any(extra[key] != meta[key] for key in
                    ("seed", "split", "layout", "target_object_id", "context_plan", "object_to_occluder")):
                    raise ValueError(f"demonstration differs from parent scene: {path}")
                candidates.append((demo["id"], path))
        for demo_id, path in candidates:
            if path.resolve() in seen:
                raise ValueError(f"duplicate demonstration: {path}")
            seen.add(path.resolve())
            expanded.append({**row, "path": str(path.resolve().relative_to(root.resolve())),
                             "parent_path": row["path"], "demonstration_id": demo_id})
    return expanded


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
    """Mean/std on valid TRAIN action frames, without decoding RGB or depth."""
    if (root / "normalization.json").is_file():
        statistics = json.loads((root / "normalization.json").read_text())
        if statistics["action_representation"] not in ("absolute_joint", "relative_joint", "relative_ee") or any(r["split"] != "train" or r["group"] != "standard" for r in rows):
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
    result = {"valid_train_frames": total}
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
    rgb = torch.from_numpy(episode["rgb"][rgb_indices].copy()).permute(3, 0, 1, 2).to(device)
    rgb = (rgb.float() / 255.0 - .45) / .225
    if "depth" in episode:  # Only explicit historical checkpoint evaluation requests depth.
        depth = torch.from_numpy(episode["depth"][indices].copy()).to(device)
        depth = ((depth.clamp(.4, 1.6) - 1.0) / .6)[None]
        rgb = torch.cat((rgb, depth), dim=0)
    return rgb[None]


def load_policy_episode(path: Path, require_wrist=False, training_cache=False, include_depth=False,
                        videos=None):
    """Load 25 Hz policy data without the large three-mask supervision array."""
    meta = json.loads((path / "episode.json").read_text())
    if meta["fps"] != 25 or meta["resolution"] != [240, 320]:
        raise ValueError(f"expected 25 Hz 240x320: {path}")
    if videos is None:
        rgb = read_rgb(path / "rgb.mp4", meta["frames"], (240, 320))
    else:
        from OATFlow.policy.loader import LeRobotVideoFrames
        rgb = LeRobotVideoFrames(videos[(path / "rgb.mp4").resolve()], meta["frames"], (240, 320))
    with np.load(path / "observation.npz") as z:
        depth = z["depth_m"] if include_depth else None
        proprio = z["joint_position"]
    with np.load(path / "supervision.npz") as z:
        action = z["expert_action"]
        valid = z["action_valid"]
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
    episode = dict(rgb=rgb, proprio=proprio, action=action, action_valid=valid,
                   query=query, valid_ends=valid_ends, events=events, name=path.name,
                   discontinuities=meta.get("action_discontinuity_frames", ()))
    if include_depth:
        if len(depth) != n:
            raise ValueError(f"depth frames disagree: {path}")
        episode["depth"] = depth
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
    ee = (path / "ee_actions.npz").is_file()
    if ee and (path / "relative_actions.npz").is_file():
        raise ValueError(f"ambiguous joint/EE action labels: {path}")
    if ee or (path / "relative_actions.npz").is_file():
        with np.load(path / ("ee_actions.npz" if ee else "relative_actions.npz")) as z:
            episode.update(relative_action=z["action"], relative_starts=z["starts"],
                           relative_valid=z["valid"], relative_anchor=z["joint_anchor" if ee else "anchor"],
                           action_representation="relative_ee" if ee else "relative_joint")
            if z["action"].shape != (len(z["starts"]), 25, 7 if ee else 6):
                raise ValueError(f"invalid action chunk shape: {path}")
        expected = np.array([end for end, active in training_samples(episode) if active])
        if not np.array_equal(expected, episode["relative_starts"]):
            raise ValueError(f"relative chunk starts differ: {path}")
        np.testing.assert_array_equal(episode["relative_anchor"], proprio[expected])
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


def load_joint_episode(path: Path, require_wrist=False, training_cache=False, include_depth=False,
                       videos=None):
    episode = load_policy_episode(path, require_wrist=require_wrist, training_cache=training_cache,
                                  include_depth=include_depth, videos=videos)
    with np.load(path / "tcow_labels.npz") as z:
        masks = z["mask"]
    if len(masks) != len(episode["rgb"]):
        raise ValueError(f"TCOW labels disagree with RGB: {path}")
    if training_cache:
        if masks.dtype != np.bool_:
            raise ValueError(f"expected binary TCOW masks: {path}")
        episode["mask_packed"] = np.packbits(masks, axis=-1)
    else:
        episode["mask"] = masks
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


def sample_plan(valid, decision: int, name: str, discontinuities=()):
    """Sample both phases at 10-frame intervals, with 25-step action targets."""
    starts, _ = chunk_starts(valid, decision, name, discontinuities)
    context = list(range(0, decision, 10))
    if context[-1] != decision - 1:
        context.append(decision - 1)
    return ([(end, False) for end in context] +
            [(int(end), True) for end in starts])


def sample_index(path: Path):
    meta = json.loads((path / "episode.json").read_text())
    with np.load(path / "supervision.npz") as z:
        valid = z["action_valid"]
    return sample_plan(valid, int(meta["decision_frames"]["t_occ"]), path.name,
                       meta.get("action_discontinuity_frames", ()))


def training_samples(episode):
    return sample_plan(episode["action_valid"], episode["events"][0], episode["name"],
                       episode.get("discontinuities", ()))


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
    if has_action and episode.get("action_representation") in ("relative_joint", "relative_ee"):
        index = int(np.searchsorted(episode["relative_starts"], end))
        if index >= len(episode["relative_starts"]) or episode["relative_starts"][index] != end:
            raise ValueError(f"missing relative chunk: {episode['name']}, {end}")
        np.testing.assert_array_equal(episode["relative_valid"][index], valid_steps[0].cpu().numpy())
        action = torch.from_numpy(episode["relative_action"][index].copy())[None].to(device)
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
