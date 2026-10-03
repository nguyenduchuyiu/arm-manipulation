"""Finalize absolute or fixed-anchor delta labels and TRAIN normalization in place."""
import json
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm


def relative_chunks(action, state, valid, decision, name):
    if not 0 < decision < len(valid) or valid[:decision].any() or not valid[decision]:
        raise ValueError(f"unexpected action boundary: {name}")
    stop = decision + int(np.flatnonzero(~valid[decision:])[0]) if (~valid[decision:]).any() else len(valid)
    if valid[stop:].any():
        raise ValueError(f"noncontiguous valid actions: {name}")
    starts = np.arange(decision, stop, 10, dtype=np.int64)
    indices = starts[:, None] + np.arange(25)
    weights = indices < stop
    absolute = action[np.minimum(indices, stop - 1)].copy()
    relative = absolute.copy()
    relative[..., :5] -= state[starts, None, :5]
    restored = relative.copy()
    restored[..., :5] += state[starts, None, :5]
    if not np.allclose(restored, absolute, rtol=0, atol=2e-7):
        raise ValueError(f"delta roundtrip failed: {name}")
    if not np.array_equal(relative[..., 5], absolute[..., 5]):
        raise ValueError(f"gripper changed: {name}")
    return dict(starts=starts, action=relative, valid=weights, anchor=state[starts].copy())


def finalize_actions(root: Path, rows, mode: str):
    if mode not in ("absolute", "delta"):
        raise ValueError(mode)
    sums = {key: np.zeros(6, np.float64) for key in ("action", "state")}
    squares = {key: np.zeros(6, np.float64) for key in sums}
    counts = dict(action=0, state=0)
    chunks_by_split = {}
    for row in tqdm(rows, desc=f"finalize {mode} actions", unit="episode", mininterval=5, file=sys.stdout):
        path = root / row["path"]
        meta = json.loads((path / "episode.json").read_text())
        with np.load(path / "supervision.npz") as z:
            action, valid = z["expert_action"], z["action_valid"]
        with np.load(path / "observation.npz") as z:
            state = z["joint_position"]
        if mode == "delta":
            chunks = relative_chunks(action, state, valid, meta["decision_frames"]["t_occ"], row["path"])
            np.savez_compressed(path / "relative_actions.npz", **chunks)
            action_values = chunks["action"][chunks["valid"]]
            chunks_by_split[row["split"]] = chunks_by_split.get(row["split"], 0) + len(chunks["starts"])
        else:
            action_values = action[valid]
        if row["split"] == "train" and row["group"] == "standard":
            for key, value in (("action", action_values), ("state", state[valid])):
                value = value.astype(np.float64)
                sums[key] += value.sum(axis=0)
                squares[key] += (value * value).sum(axis=0)
                counts[key] += len(value)
    if not all(counts.values()):
        raise ValueError("no valid standard TRAIN actions for normalization")
    representation = "relative_joint" if mode == "delta" else "absolute_joint"
    statistics = {"valid_train_frames": counts["state"], "valid_train_chunk_steps": counts["action"],
                  "action_representation": representation, "normalization_split": "standard/train",
                  "horizon": 25, "stride": 10}
    for key in sums:
        mean = sums[key] / counts[key]
        std = np.sqrt(np.maximum(squares[key] / counts[key] - mean * mean, 0))
        std = np.where(std < (1e-8 if mode == "delta" else 1e-3), 1., std)
        statistics[key + "_mean"], statistics[key + "_std"] = mean.tolist(), std.tolist()
    (root / "normalization.json").write_text(json.dumps(statistics, indent=2) + "\n")
    metadata = {"episodes": len(rows), "action_mode": mode, "action_representation": representation,
                "horizon": 25, "stride": 10, "fps": 25, "wrist_camera": True,
                "chunks_by_split": chunks_by_split,
                "anchor": "observed joint_position at chunk start; fixed for 25 actions",
                "gripper": "absolute 0 closed, 1 open", "storage": "self-contained; absolute expert labels retained"}
    (root / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"event": "dataset_ready", **metadata}), flush=True)
