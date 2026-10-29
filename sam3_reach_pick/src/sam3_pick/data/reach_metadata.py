"""Rebuild voxel-disjoint splits, fixed evaluation IDs and train-only normalization."""

import json
import math
import numpy as np
from tqdm import tqdm
from sam3_pick.paths import ASSETS
from sam3_pick.data.reach_clouds import DATA, REFERENCE, write


def main():
    REFERENCE.mkdir(parents=True, exist_ok=True)
    episode = np.load(DATA / "chunk_trajectory_index.npy", mmap_mode="r")
    centers = np.load(DATA / "target_world_m.npy", mmap_mode="r")
    rng = np.random.default_rng(78)
    order = rng.permutation(len(centers))
    held = math.ceil(len(order) * 0.1)
    split = dict(
        val_trajectories=np.sort(order[:held]),
        test_trajectories=np.sort(order[held : 2 * held]),
        train_trajectories=np.sort(order[2 * held :]),
    )
    for name in ["train", "val", "test"]:
        split[name + "_chunks"] = np.flatnonzero(np.isin(episode, split[name + "_trajectories"]))
    np.savez(REFERENCE / "split.npz", **split)
    rng = np.random.default_rng(79)
    evaluation = {}
    for name, pool, n in [
        ("val_chunks", "val_chunks", 2048),
        ("val_episodes", "val_trajectories", 32),
        ("test_chunks", "test_chunks", 8192),
        ("test_episodes", "test_trajectories", 128),
        ("train_episodes", "train_trajectories", 32),
    ]:
        assert len(split[pool]) >= n, "Full voxel dataset required for training metadata"
        evaluation[name] = rng.choice(split[pool], n, replace=False)
    np.savez(REFERENCE / "evaluation_indices.npz", **evaluation)
    actions = np.load(DATA / "actions.npy", mmap_mode="r")
    mask = np.load(DATA / "action_mask.npy", mmap_mode="r")
    joints = np.load(DATA / "joint_states/state_joint_position.npy", mmap_mode="r")
    base = np.array(
        json.loads((ASSETS / "reach_metadata/config.json").read_text())["base_origin_world_m"]
    )
    sums = {k: np.zeros(d, np.float64) for k, d in [("state", 5), ("action", 9), ("xyz", 3)]}
    squares = {k: v.copy() for k, v in sums.items()}
    counts = dict(state=0, action=0, xyz=0)
    train = split["train_chunks"]
    for start in tqdm(
        range(0, len(train), 8192), desc="Reach train-only statistics", mininterval=5
    ):
        ids = train[start : start + 8192]
        values = dict(
            state=np.asarray(joints[ids, :5], dtype=np.float64),
            action=np.asarray(actions[ids, :, :9], dtype=np.float64)[mask[ids]],
            xyz=np.asarray(centers[episode[ids]], dtype=np.float64) - base,
        )
        for name, value in values.items():
            sums[name] += value.sum(0)
            squares[name] += np.square(value).sum(0)
            counts[name] += len(value)
    stats = dict(
        base_origin_world_m=base.tolist(),
        source="Current generated voxel data; train chunks only",
        split_seed=78,
        evaluation_seed=79,
    )
    for name in sums:
        mean = sums[name] / counts[name]
        std = np.sqrt(np.maximum(squares[name] / counts[name] - mean**2, 0))
        if name in ["state", "action"]:
            mean = np.r_[mean, 0.0]
            std = np.r_[std, 1.0]
        assert (std > 0).all()
        stats[name + "_mean"] = mean.tolist()
        stats[name + "_std"] = std.tolist()
    write(REFERENCE / "normalization.json", stats)
    print("Rebuilt splits/evaluation indices and train-only statistics", flush=True)


if __name__ == "__main__":
    main()
