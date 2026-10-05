"""Write joint/EE action chunks and fit normalization on standard TRAIN demos."""
import json
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm


def chunk_starts(valid, decision, name, discontinuities=()):
    if not 0 < decision < len(valid) or valid[:decision].any() or not valid[decision]:
        raise ValueError(f"unexpected action boundary: {name}")
    stop = decision + int(np.flatnonzero(~valid[decision:])[0]) if (~valid[decision:]).any() else len(valid)
    if valid[stop:].any():
        raise ValueError(f"noncontiguous valid actions: {name}")
    starts = np.arange(decision, stop, 10, dtype=np.int64)
    for boundary in discontinuities:
        if not decision <= boundary < stop:
            raise ValueError(f"disturbance outside valid actions: {name}")
        starts = starts[~((starts < boundary) & (starts + 25 > boundary))]
    return starts, stop


def relative_chunks(action, state, valid, decision, name, discontinuities=()):
    starts, stop = chunk_starts(valid, decision, name, discontinuities)
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


def save_action_chunks(path, action, state, valid, metadata, mode, kinematics=None):
    decision = metadata["decision_frames"]["t_occ"]
    boundaries = metadata.get("action_discontinuity_frames", ())
    if mode == "ee":
        from OATFlow.dataset.ee_actions import ee_chunks
        chunks = ee_chunks(kinematics, action, state, valid, decision, path.name, boundaries)
        filename = "ee_actions.npz"
    elif mode == "delta":
        chunks = relative_chunks(action, state, valid, decision, path.name, boundaries)
        filename = "relative_actions.npz"
    else:
        raise ValueError(mode)
    np.savez_compressed(path / filename, **chunks)
    return chunks


def finalize_actions(root: Path, rows, mode: str):
    if mode not in ("absolute", "delta", "ee"):
        raise ValueError(mode)
    kinematics = None
    if mode == "ee":
        from OATFlow.dataset.ee_actions import EEKinematics
        kinematics = EEKinematics()
    sums = {"action": np.zeros(7 if mode == "ee" else 6, np.float64),
            "state": np.zeros(6, np.float64)}
    squares = {key: np.zeros_like(value) for key, value in sums.items()}
    counts = dict(action=0, state=0)
    chunks_by_split = {}
    for row in tqdm(rows, desc=f"finalize {mode} actions", unit="episode", mininterval=5, file=sys.stdout):
        path = root / row["path"]
        meta = json.loads((path / "episode.json").read_text())
        with np.load(path / "supervision.npz") as z:
            action, valid = z["expert_action"], z["action_valid"]
        with np.load(path / "observation.npz") as z:
            state = z["joint_position"]
        if mode in ("delta", "ee"):
            chunks = save_action_chunks(path, action, state, valid, meta, mode, kinematics)
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
    representation = {"absolute": "absolute_joint", "delta": "relative_joint", "ee": "relative_ee"}[mode]
    statistics = {"valid_train_frames": counts["state"], "valid_train_chunk_steps": counts["action"],
                  "action_representation": representation, "normalization_split": "standard/train",
                  "horizon": 25, "stride": 10, "action_dim": 7 if mode == "ee" else 6}
    for key in sums:
        mean = sums[key] / counts[key]
        std = np.sqrt(np.maximum(squares[key] / counts[key] - mean * mean, 0))
        std = np.where(std < (1e-8 if mode != "absolute" else 1e-3), 1., std)
        statistics[key + "_mean"], statistics[key + "_std"] = mean.tolist(), std.tolist()
    (root / "normalization.json").write_text(json.dumps(statistics, indent=2) + "\n")
    metadata = {"episodes": sum(row.get("demonstration_id", "base") == "base" for row in rows),
                "demonstrations": len(rows), "action_mode": mode, "action_representation": representation,
                "horizon": 25, "stride": 10, "fps": 25, "wrist_camera": True,
                "chunks_by_split": chunks_by_split,
                "anchor": ("observed TCP pose at chunk start; fixed for 25 actions" if mode == "ee" else
                           "observed joint_position at chunk start; fixed for 25 actions"),
                "gripper": "absolute 0 closed, 1 open", "storage": "self-contained; absolute expert labels retained"}
    if mode == "ee":
        metadata.update(action_dim=7, state_dim=6,
                        dimensions=["dx_m", "dy_m", "dz_m", "rx_rad", "ry_rad", "rz_rad", "gripper_absolute"],
                        frame="world; R_command = Exp(delta_rotation) @ R_observed_at_chunk_start",
                        command="FK of executed actuator setpoints, including gravity compensation")
    (root / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"event": "dataset_ready", **metadata}), flush=True)
