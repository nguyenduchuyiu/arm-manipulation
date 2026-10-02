"""Create 25-action relative-joint chunks, retaining absolute gripper labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm

from memory_occlusion.experiments.tcow_joint_flow.data import sample_plan
from memory_occlusion.experiments.tcow_joint_flow.probe_geometry import selected_npz_frames


def relative_chunks(action, state, valid, decision, name):
    starts = np.array([t for t, active in sample_plan(valid, decision, name) if active], np.int64)
    indices = starts[:, None] + np.arange(25)[None]
    inside = indices < len(valid)
    safe = np.minimum(indices, len(valid) - 1)
    weights = inside & valid[safe]
    # Match training's tail padding; excluded values do not fit normalization.
    last = starts + weights.sum(axis=1) - 1
    indices = np.minimum(indices, last[:, None])
    absolute = action[indices].copy()
    relative = absolute.copy()
    relative[..., :5] -= state[starts, None, :5]
    restored = relative.copy()
    restored[..., :5] += state[starts, None, :5]
    if not np.allclose(restored, absolute, rtol=0, atol=2e-7):
        raise ValueError(f"relative roundtrip failed: {name}")
    if not np.array_equal(relative[..., 5], absolute[..., 5]):
        raise ValueError(f"gripper changed: {name}")
    return dict(starts=starts, action=relative, valid=weights, anchor=state[starts].copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = (args.source / "manifest.jsonl").read_bytes()
    rows = [json.loads(line) for line in manifest.decode().splitlines()]
    if len(rows) != 880:
        raise ValueError("expected complete 880-episode source dataset")
    args.output.mkdir(parents=True)
    source = args.source.resolve()
    sums = {key: np.zeros(6, np.float64) for key in ("action", "state")}
    squares = {key: np.zeros(6, np.float64) for key in sums}
    counts = dict(action=0, state=0)
    chunks_by_split = {}
    max_error = 0.
    for row in tqdm(rows, desc="convert relative actions", unit="episode", mininterval=5, file=sys.stdout):
        path, output = source / row["path"], args.output / row["path"]
        meta = json.loads((path / "episode.json").read_text())
        with np.load(path / "supervision.npz") as z:
            action, valid = z["expert_action"], z["action_valid"]
        frames = np.arange(len(valid), dtype=np.int64)
        state = selected_npz_frames(path / "observation.npz", "joint_position", frames)
        chunks = relative_chunks(action, state, valid, meta["decision_frames"]["t_occ"], row["path"])
        output.mkdir(parents=True)
        for name in ("rgb.mp4", "observation.npz", "supervision.npz", "tcow_labels.npz",
                     "episode.json", "query_mask.png", "debug_poses.npz"):
            if not (path / name).is_file():
                raise FileNotFoundError(path / name)
            (output / name).symlink_to(path / name)
        np.savez_compressed(output / "relative_actions.npz", **chunks)
        restored = chunks["action"].copy()
        restored[..., :5] += chunks["anchor"][:, None, :5]
        indices = np.minimum(chunks["starts"][:, None] + np.arange(25), len(action) - 1)
        max_error = max(max_error, float(np.abs(restored - action[indices])[chunks["valid"]].max()))
        chunks_by_split[row["split"]] = chunks_by_split.get(row["split"], 0) + len(chunks["starts"])
        if row["split"] == "train":
            for key, value in (("action", chunks["action"][chunks["valid"]]), ("state", state[valid])):
                value = value.astype(np.float64)
                sums[key] += value.sum(axis=0)
                squares[key] += (value * value).sum(axis=0)
                counts[key] += len(value)
    statistics = {"valid_train_frames": counts["state"], "valid_train_chunk_steps": counts["action"],
                  "action_representation": "relative_joint", "horizon": 25, "stride": 10}
    for key in sums:
        mean = sums[key] / counts[key]
        std = np.sqrt(np.maximum(squares[key] / counts[key] - mean * mean, 0))
        std = np.where(std < 1e-8, 1., std)
        statistics[key + "_mean"], statistics[key + "_std"] = mean.tolist(), std.tolist()
    (args.output / "normalization.json").write_text(json.dumps(statistics, indent=2) + "\n")
    (args.output / "manifest.jsonl").write_bytes(manifest)
    metadata = {"source": str(source), "source_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "episodes": len(rows), "chunks_by_split": chunks_by_split,
                "action_representation": "relative_joint", "horizon": 25, "stride": 10,
                "anchor": "observed joint_position at chunk start; fixed for all 25 actions",
                "gripper": "absolute 0 closed, 1 open", "storage": "own action chunks; linked read-only RGB-D/labels",
                "normalization_split": "train", "roundtrip_max_error": max_error}
    (args.output / "dataset.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"event": "relative_dataset_ready", **metadata, "normalization": statistics}), flush=True)


if __name__ == "__main__":
    main()
