"""Check short cover-drop/target-lift absolute labels before training."""
import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from tqdm.auto import tqdm

from OATFlow.dataset.actions import chunk_starts
from OATFlow.environment.env import MemoryOcclusionEnv
from OATFlow.environment.success import cover_deposited, target_lifted


def audit(root, demos_per_task):
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    if len(rows) != 12 * demos_per_task:
        raise ValueError("incomplete demo count")
    counts = np.bincount([r["task_id"] for r in rows], minlength=12)
    if not np.all(counts == demos_per_task) or len({r["seed"] for r in rows}) != len(rows):
        raise ValueError("unbalanced tasks or duplicate scene seeds")
    statistics = json.loads((root / "normalization.json").read_text())
    if statistics["action_representation"] != "absolute_joint" or statistics["action_dim"] != 6:
        raise ValueError("absolute joint labels required")
    env = MemoryOcclusionEnv(resolution=160)
    jaws = [env.model.jnt_qposadr[env.model.joint(f"{s}_jaw_slide_joint").id]
            for s in ("left", "right")]
    chunks, valid_frames, still_count, transitions = 0, 0, 0, 0
    max_error, longest_still = 0., 0
    try:
        for row in tqdm(rows, desc="audit short absolute task", unit="demo", file=sys.stdout):
            path = root / row["path"]
            meta = json.loads((path / "episode.json").read_text())
            if any(meta[key] != row[key] for key in ("task_id", "target_id", "demo_id", "seed", "frames")):
                raise ValueError(f"manifest disagrees with episode: {path}")
            if not meta.get("target_lifted") or not meta.get("cover_deposited") or not meta["success"]:
                raise ValueError(f"incomplete cover drop/target lift: {path}")
            with np.load(path / "supervision.npz") as z:
                absolute, valid, phase = z["expert_action"], z["action_valid"], z["phase"]
            with np.load(path / "observation.npz") as z:
                observed = z["joint_position"]
            with np.load(path / "absolute_actions.npz") as z:
                starts, actions, weights, anchor = z["starts"], z["action"], z["valid"], z["anchor"]
            expected, stop = chunk_starts(valid, meta["decision_frames"]["t_occ"], path.name)
            if not np.array_equal(starts, expected):
                raise ValueError(f"missing stride1 expert frames: {path}")
            if actions.shape != (len(starts), 25, 6) or weights.shape != (len(starts), 25):
                raise ValueError(f"bad action shape: {path}")
            if not np.array_equal(weights, starts[:, None] + np.arange(25) < stop):
                raise ValueError(f"wrong terminal action mask: {path}")
            if not np.array_equal(anchor, observed[starts]):
                raise ValueError(f"wrong observed joint anchor: {path}")
            indices = np.minimum(starts[:, None] + np.arange(25), stop - 1)
            error = float(np.abs(actions - absolute[indices]).max())
            if error != 0 or not np.array_equal(actions[..., 5], absolute[indices, 5]):
                raise ValueError(f"joint/gripper roundtrip failed: {path}")
            for stage in ("approach", "engage", "close", "lift", "hold"):
                mask = phase == meta["target_object_id"] + "_" + stage
                if not mask.any() or not valid[mask].all():
                    raise ValueError(f"missing target {stage} labels: {path}")
            for stage in ("transport", "release", "settle"):
                if not (phase == meta["selected_cover"] + "_" + stage).any():
                    raise ValueError(f"missing cover {stage}: {path}")
            if any(np.char.endswith(phase, '_' + s).any() for s in ('lower', 'home', 'retreat')):
                raise ValueError(f"obsolete long phases in short data: {path}")
            if any((phase == meta['target_object_id'] + '_' + s).any() for s in ('transport', 'release', 'settle')):
                raise ValueError(f"target placement present in lift-only task: {path}")
            with np.load(path / "sim_state.npz") as z:
                qpos, qvel, initial_qpos = z["qpos"], z["qvel"], z["initial_qpos"]
            mujoco.mj_resetData(env.model, env.data)
            env.data.qpos[:] = qpos[-1]
            env.data.qvel[:] = qvel[-1]
            mujoco.mj_forward(env.model, env.data)
            target = meta["target_object_id"]
            if meta.get("bilateral_lift_streak_frames", {}).get(target, 0) < 10:
                raise ValueError(f"target lift lacks ten bilateral-contact frames: {path}")
            initial_z = initial_qpos[env.target_qadr[target] + 2]
            if not cover_deposited(env.model, env.data, meta["selected_cover"]) or not target_lifted(env.model, env.data, target, initial_z):
                raise ValueError(f"terminal cover drop/target lift failed: {path}")
            delta = np.abs(np.diff(qpos, axis=0))
            eligible = valid[:-1] & valid[1:]
            still = (delta[:, env.robot_qpos_addresses[:5]].max(1) <= np.deg2rad(.01))
            still &= delta[:, jaws].max(1) <= .00001
            still &= eligible
            current = 0
            for value in still:
                current = current + 1 if value else 0
                longest_still = max(longest_still, current + 1)
            still_count += int(still.sum())
            transitions += int(eligible.sum())
            chunks += len(starts)
            valid_frames += int(valid.sum())
            max_error = max(max_error, error)
        for task_id in range(12):
            targets = np.unique([r["target_id"] for r in rows if r["task_id"] == task_id])
            if len(targets) != 2 or any(sum(r["task_id"] == task_id and r["target_id"] == target
                                          for r in rows) != demos_per_task // 2 for target in targets):
                raise ValueError("unbalanced target identities")
    finally:
        env.close()
    result = dict(status="passed", demos=len(rows), per_task=counts.tolist(), chunks=chunks,
                  valid_frames=valid_frames, action_representation="absolute_joint",
                  action_dim=6, max_command_error=max_error, all_covers_deposited=True, all_targets_lifted=True,
                  stationary_fraction=still_count / transitions, max_stationary_observations=longest_still)
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--demos-per-task", type=int, default=50)
    args = parser.parse_args()
    audit(args.data, args.demos_per_task)
