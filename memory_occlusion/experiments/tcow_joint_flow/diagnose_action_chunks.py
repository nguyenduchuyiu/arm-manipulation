"""Compare 25-step expert-state predictions with raw closed-loop policy chunks."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

from memory_occlusion.experiments.tcow_joint_flow.data import load_policy_episode, policy_clip
from memory_occlusion.experiments.tcow_joint_flow.model import expand_depth_channel, restore_joint_model
from memory_occlusion.experiments.tcow_joint_flow.tcow import Seeker


def mean(values):
    return np.mean(np.asarray(values), axis=0).tolist()


@torch.inference_mode()
def offline(model, path, noise, device):
    episode = load_policy_episode(path)
    meta = json.loads((path / "episode.json").read_text())
    decision = episode["events"][0]
    available = set(map(int, episode["valid_ends"]))
    ends = [frame for frame in range(decision, int(episode["valid_ends"][-1]) + 1, 10)
            if frame in available]
    rows = []
    for end in tqdm(ends, desc=f"offline {path.name}", unit="chunk",
                    mininterval=5, file=sys.stdout):
        rgbd, query, state, truth = policy_clip(episode, end, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            model._latent = None
            model.tcow(rgbd, query)
            predicted = model.flow.sample(model._latent, state, noise)
        predicted = predicted[0].float().cpu().numpy()
        truth = truth[0].float().cpu().numpy()
        state = state[0].float().cpu().numpy()
        executed = predicted.copy()
        executed[:, :5] = np.clip(executed[:, :5], -1, 1)
        executed[:, 5] = (executed[:, 5] >= .5).astype(np.float32)
        error = np.abs(predicted - truth)
        truth_closed = truth[:, 5] < .5
        predicted_closed = predicted[:, 5] < .5
        rows.append({"frame": end, "phase": "cover" if end < meta["decision_frames"]["t_obj"] else "object",
                     "raw_mae_25": float(error.mean()),
                     "raw_arm_mae_25": float(error[:, :5].mean()),
                     "raw_mae_first10": float(error[:10].mean()),
                     "executed_mae_first10": float(np.abs(executed[:10] - truth[:10]).mean()),
                     "hold_mae_25": float(np.abs(state[None] - truth).mean()),
                     "hold_arm_mae_25": float(np.abs(state[None, :5] - truth[:, :5]).mean()),
                     "expert_gripper_closed": int(truth_closed.sum()),
                     "predicted_gripper_closed": int(predicted_closed.sum()),
                     "correct_gripper_closed": int((truth_closed & predicted_closed).sum()),
                     "raw_mae_per_joint": error.mean(axis=0).tolist(),
                     "state": state.tolist(),
                     "raw_action_0": predicted[0].tolist(),
                     "raw_action_9": predicted[9].tolist(),
                     "raw_action_24": predicted[24].tolist(),
                     "expert_action_0": truth[0].tolist(),
                     "expert_action_9": truth[9].tolist(),
                     "expert_action_24": truth[24].tolist()})
    if not rows:
        raise ValueError(f"no full 25-step action chunks: {path}")
    def phase_scores(subset):
        closed = sum(r["expert_gripper_closed"] for r in subset)
        total = 25 * len(subset)
        return {"chunks": len(subset),
                "raw_mae_25": mean([r["raw_mae_25"] for r in subset]) if subset else None,
                "raw_arm_mae_25": mean([r["raw_arm_mae_25"] for r in subset]) if subset else None,
                "raw_mae_first10": mean([r["raw_mae_first10"] for r in subset]) if subset else None,
                "hold_mae_25": mean([r["hold_mae_25"] for r in subset]) if subset else None,
                "hold_arm_mae_25": mean([r["hold_arm_mae_25"] for r in subset]) if subset else None,
                "expert_gripper_closed_fraction": closed / total if total else None,
                "predicted_gripper_closed_fraction":
                    sum(r["predicted_gripper_closed"] for r in subset) / total if total else None,
                "gripper_close_recall":
                    sum(r["correct_gripper_closed"] for r in subset) / closed if closed else None}

    summary = {"episode": path.name, "seed": meta["seed"], "target": meta["target_object_id"],
               "decision_frame": decision, "object_phase_frame": meta["decision_frames"]["t_obj"],
               "chunks": len(rows), "raw_mae_25": mean([r["raw_mae_25"] for r in rows]),
               "raw_arm_mae_25": mean([r["raw_arm_mae_25"] for r in rows]),
               "raw_mae_first10": mean([r["raw_mae_first10"] for r in rows]),
               "executed_mae_first10": mean([r["executed_mae_first10"] for r in rows]),
               "hold_mae_25": mean([r["hold_mae_25"] for r in rows]),
               "hold_arm_mae_25": mean([r["hold_arm_mae_25"] for r in rows]),
               "raw_mae_per_joint": mean([r["raw_mae_per_joint"] for r in rows]),
               "overall": phase_scores(rows), "by_phase": {}}
    for phase in ("cover", "object"):
        subset = [r for r in rows if r["phase"] == phase]
        summary["by_phase"][phase] = phase_scores(subset)
    summary["examples"] = [rows[index] for index in sorted({0, len(rows) // 2, len(rows) - 1})]
    return summary


def online(path):
    chunks = [json.loads(line) for line in (path / "policy_chunks.jsonl").read_text().splitlines()]
    trace = [json.loads(line) for line in (path / "trace.jsonl").read_text().splitlines()]
    if not chunks or not trace:
        raise ValueError(f"empty closed-loop trace: {path}")
    raw = np.asarray([row["action_25_raw"] for row in chunks], dtype=np.float32)
    states = np.asarray([row["proprio"] for row in chunks], dtype=np.float32)
    executed = np.asarray([row["action"] for row in trace], dtype=np.float32)
    observed = np.asarray([row["proprio"] for row in trace], dtype=np.float32)
    arm_clipped = np.abs(raw[:, :10, :5]) > 1
    gripper_closed = raw[:, :, 5] < .5
    out = {"rollout": path.name, "policy_chunks": len(chunks), "executed_steps": len(trace),
           "raw_action_min": raw.min(axis=(0, 1)).tolist(),
           "raw_action_max": raw.max(axis=(0, 1)).tolist(),
           "raw_action_mean": raw.mean(axis=(0, 1)).tolist(),
           "raw_action_std": raw.std(axis=(0, 1)).tolist(),
           "arm_clip_fraction_first10": float(arm_clipped.mean()),
           "predicted_gripper_closed_fraction_first10": float(gripper_closed[:, :10].mean()),
           "predicted_gripper_closed_fraction_later15": float(gripper_closed[:, 10:].mean()),
           "predicted_gripper_closed_count_by_offset": gripper_closed.sum(axis=0).tolist(),
           "raw_gripper_open_fraction_first10": float((raw[:, :10, 5] >= .5).mean()),
           "executed_gripper_open_fraction": float((executed[:, 5] >= .5).mean()),
           "mean_abs_first_action_minus_state": float(np.abs(raw[:, 0] - states).mean()),
           "observed_proprio_start": observed[0].tolist(),
           "observed_proprio_end": observed[-1].tolist(),
           "observed_proprio_range": np.ptp(observed, axis=0).tolist(),
           "first_chunk": chunks[0], "middle_chunk": chunks[len(chunks) // 2],
           "last_chunk": chunks[-1]}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, nargs="+", required=True)
    p.add_argument("--rollouts", type=Path, nargs="+")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    args = p.parse_args()
    if args.rollouts and len(args.episodes) != len(args.rollouts):
        raise ValueError("use one rollout directory per expert episode")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    config = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    seeker_args = dict(config["seeker_args"])
    seeker_args["tracker_pretrained"] = False
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    tcow = expand_depth_channel(Seeker(logging.getLogger("tcow"), **seeker_args))
    model = restore_joint_model(tcow, saved).to(args.device).eval()
    noise = model.flow.sample_noise(1, "cpu", torch.Generator().manual_seed(0)).to(args.device)
    results = []
    for index, episode_path in enumerate(args.episodes):
        result = {"offline": offline(model, episode_path, noise, args.device)}
        if args.rollouts:
            result["online"] = online(args.rollouts[index])
        results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"checkpoint_step": saved["step"], "episodes": results}, indent=2) + "\n")
    for result in results:
        print(json.dumps({"episode": result["offline"]["episode"],
                          "offline_raw_mae_25": result["offline"]["raw_mae_25"],
                          "offline_by_phase": result["offline"]["by_phase"],
                          "online_proprio_range": result["online"]["observed_proprio_range"] if "online" in result else None,
                          "online_arm_clip_fraction": result["online"]["arm_clip_fraction_first10"] if "online" in result else None}),
              flush=True)


if __name__ == "__main__":
    main()
