"""Write absolute joint action chunks and fit normalization on standard TRAIN demos."""
import json
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm


def chunk_starts(valid, decision, name):
    """Start a chunk at every valid expert frame, including short tails."""
    if not 0 < decision < len(valid) or valid[:decision].any() or not valid[decision]:
        raise ValueError(f"unexpected action boundary: {name}")
    stop = decision + int(np.flatnonzero(~valid[decision:])[0]) if (~valid[decision:]).any() else len(valid)
    if valid[stop:].any():
        raise ValueError(f"noncontiguous valid actions: {name}")
    return np.arange(decision, stop, dtype=np.int64), stop


def save_action_chunks(path, action, state, valid, metadata):
    starts, stop = chunk_starts(valid, metadata["decision_frames"]["t_occ"], path.name)
    indices = starts[:, None] + np.arange(25)
    chunks = dict(starts=starts, action=action[np.minimum(indices, stop - 1)].copy(),
                  valid=indices < stop, anchor=state[starts].copy())
    np.savez_compressed(path / "absolute_actions.npz", **chunks)
    return chunks


def finalize_actions(root: Path, rows):
    sums = {key: np.zeros(6, np.float64) for key in ("action", "state")}
    squares = {key: np.zeros_like(value) for key, value in sums.items()}
    counts = dict(action=0, state=0)
    chunks_by_split = {}
    for row in tqdm(rows, desc="finalize absolute actions", unit="episode", mininterval=5, file=sys.stdout):
        path = root / row["path"]
        meta = json.loads((path / "episode.json").read_text())
        with np.load(path / "supervision.npz") as z:
            action, valid = z["expert_action"], z["action_valid"]
        with np.load(path / "observation.npz") as z:
            state = z["joint_position"]
        chunks = save_action_chunks(path, action, state, valid, meta)
        chunks_by_split[row["split"]] = chunks_by_split.get(row["split"], 0) + len(chunks["starts"])
        if row["split"] == "train" and row["group"] == "standard":
            for key, value in (("action", chunks["action"][chunks["valid"]]), ("state", state[valid])):
                value = value.astype(np.float64)
                sums[key] += value.sum(axis=0)
                squares[key] += (value * value).sum(axis=0)
                counts[key] += len(value)
    if not all(counts.values()):
        raise ValueError("no valid standard TRAIN actions for normalization")
    statistics = dict(valid_train_frames=counts["state"], valid_train_chunk_steps=counts["action"],
                      action_representation="absolute_joint", normalization_split="standard/train",
                      horizon=25, stride=1, action_dim=6)
    for key in sums:
        mean = sums[key] / counts[key]
        std = np.sqrt(np.maximum(squares[key] / counts[key] - mean * mean, 0))
        std = np.where(std < 1e-8, 1., std)
        statistics[key + "_mean"], statistics[key + "_std"] = mean.tolist(), std.tolist()
    (root / "normalization.json").write_text(json.dumps(statistics, indent=2) + "\n")
    metadata = dict(episodes=len(rows), demonstrations=len(rows), action_representation="absolute_joint",
                    horizon=25, stride=1, fps=25, wrist_camera=True, action_dim=6,
                    chunks_by_split=chunks_by_split, anchor="observed joints for proprioception only",
                    gripper="absolute 0 closed, 1 open")
    (root / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"event": "dataset_ready", **metadata}), flush=True)
