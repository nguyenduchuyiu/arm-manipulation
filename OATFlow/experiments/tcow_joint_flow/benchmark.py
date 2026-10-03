"""Held-out joint policy benchmark at cover choice, recovery, and valid action phases."""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

from OATFlow.experiments.tcow_joint_flow.data import clip, load_joint_episode
from OATFlow.experiments.tcow_joint_flow.model import expand_depth_channel, restore_joint_model
from OATFlow.experiments.tcow_joint_flow.tcow import Seeker
from OATFlow.experiments.tcow_joint_flow.wrist import wrist_tensor


def iou(prediction, truth):
    intersection = np.logical_and(prediction, truth).sum()
    union = np.logical_or(prediction, truth).sum()
    return float(intersection / union) if union else None


def evaluate_episode(model, path, row):
    use_wrist = getattr(model.flow, "wrist_enabled", False)
    episode = load_joint_episode(path, require_wrist=use_wrist)
    meta = json.loads((path / "episode.json").read_text())
    decision = int(meta["decision_frames"]["t_occ"]) - 1
    recovery = min(int(meta["semantic_transition_frames"]["cover_released"]) + 25,
                   len(episode["rgb"]) - 26)
    parts = np.array_split(episode["valid_ends"], 4)
    action_ends = [int(part[len(part) // 2]) for part in parts]
    ends = [decision, recovery, *action_ends]
    samples = [clip(episode, end, "cuda") for end in ends]
    rgbd, query, truth, proprio, action = (torch.cat(items, dim=0)
                                            for items in zip(*samples))
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model._latent = None
        logits, _ = model.tcow(rgbd, query)
        latent = model._latent
        if latent.shape != (6, 300, 768):
            raise RuntimeError(f"unexpected TCOW latent shape: {latent.shape}")
        actions = action[2:]
        states = proprio[2:]
        generator = torch.Generator(device="cuda").manual_seed(
            int(row["seed"]) * 100 + sum(map(ord, row["target"])))
        noise = model.flow.sample_noise(len(actions), "cuda", generator)
        if use_wrist:
            wrist = torch.cat([wrist_tensor(episode["wrist_rgb"][end], "cuda") for end in action_ends])
            estimate = model.flow.sample(latent[2:], states, noise, wrist=wrist)
        else:
            estimate = model.flow.sample(latent[2:], states, noise)
    gt_action = actions.float()
    predicted_action = estimate.float()
    hold = states.float()[:, None, :].expand_as(gt_action)
    error = (predicted_action - gt_action).abs().cpu().numpy()
    hold_error = (hold - gt_action).abs().cpu().numpy()
    predicted_masks = (logits[:2, :, -1].float() > 0).cpu().numpy()
    gt_masks = truth[:2, :, -1].bool().cpu().numpy()
    decision_iou = [iou(predicted_masks[0, i], gt_masks[0, i]) for i in range(3)]
    recovery_iou = [iou(predicted_masks[1, i], gt_masks[1, i]) for i in range(3)]
    result = {"episode": row["path"], "seed": row["seed"], "target": row["target"],
              "group": row["group"], "split": row["split"],
              "cover": row["cover"], "swaps": row["swaps"],
              "is_held_out_query": row["is_held_out_query"],
              "decision_frame": decision, "recovery_frame": recovery,
              "decision_iou": decision_iou, "recovery_iou": recovery_iou,
              "decision_gt_pixels": gt_masks[0].sum(axis=(-1, -2)).tolist(),
              "decision_pred_pixels": predicted_masks[0].sum(axis=(-1, -2)).tolist(),
              "action_ends": action_ends,
              "action_mae": error.mean(axis=(1, 2)).tolist(),
              "hold_mae": hold_error.mean(axis=(1, 2)).tolist(),
              "action_mae_per_joint": error.mean(axis=(0, 1)).tolist()}
    return result, predicted_masks[0, 2], gt_masks[0, 2]


def mean_present(rows, key, channel):
    values = [row[key][channel] for row in rows if row[key][channel] is not None]
    return float(np.mean(values)) if values else None


def summary(rows):
    return {"episodes": len(rows),
            "decision_target_iou": mean_present(rows, "decision_iou", 0),
            "decision_occluder_iou": mean_present(rows, "decision_iou", 1),
            "decision_container_iou": mean_present(rows, "decision_iou", 2),
            "decision_occluder_present": sum(row["decision_iou"][1] is not None for row in rows),
            "decision_container_present": sum(row["decision_iou"][2] is not None for row in rows),
            "recovered_target_iou": mean_present(rows, "recovery_iou", 0),
            "action_mae": float(np.mean([row["action_mae"] for row in rows])),
            "hold_mae": float(np.mean([row["hold_mae"] for row in rows])),
            "action_mae_by_phase": np.mean([row["action_mae"] for row in rows], axis=0).tolist(),
            "hold_mae_by_phase": np.mean([row["hold_mae"] for row in rows], axis=0).tolist(),
            "action_mae_per_joint": np.mean([row["action_mae_per_joint"] for row in rows], axis=0).tolist(),
            "cover_choice_accuracy": float(np.mean([row["cover_choice_correct"] for row in rows]))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-episodes-per-split", type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    config = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    seeker_args = dict(config["seeker_args"])
    seeker_args["tracker_pretrained"] = False
    tcow = expand_depth_channel(Seeker(logging.getLogger("tcow"), **seeker_args))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model = restore_joint_model(tcow, saved).cuda().eval()
    manifest = [json.loads(line) for line in (args.data / "manifest.jsonl").read_text().splitlines()]
    selected = [row for row in manifest if row["split"] == "test"]
    if len(selected) != 160:
        raise ValueError(f"expected 160 test episodes, got {len(selected)}")
    by_group = defaultdict(list)
    for row in selected:
        by_group[row["group"]].append(row)
    rows = []
    print(json.dumps({"event": "start", "checkpoint_step": saved["step"],
                      "episodes": len(selected), "splits": {k: len(v) for k, v in by_group.items()}}), flush=True)
    for group, subset in by_group.items():
        if args.max_episodes_per_split:
            subset = subset[:args.max_episodes_per_split]
        seed_groups = defaultdict(list)
        for row in subset:
            seed_groups[row["seed"]].append(row)
        for seed, scene_rows in tqdm(seed_groups.items(), desc=f"benchmark {group}",
                                     unit="scene", mininterval=5, file=sys.stdout):
            scene_results = []
            for row in scene_rows:
                result, prediction, own_truth = evaluate_episode(
                    model, args.data / row["path"], row)
                scene_results.append((result, prediction, own_truth))
            for result, prediction, own_truth in scene_results:
                opposite = next((other_truth for other, _, other_truth in scene_results
                                 if other["cover"] != result["cover"]), None)
                if opposite is None:
                    if args.max_episodes_per_split:
                        result["other_cover_iou"] = None
                        result["cover_choice_correct"] = None
                    else:
                        raise RuntimeError(f"missing opposite cover for scene {seed}")
                else:
                    result["other_cover_iou"] = iou(prediction, opposite)
                    own_iou = result["decision_iou"][2]
                    result["cover_choice_correct"] = own_iou > result["other_cover_iou"]
                rows.append(result)
            (args.output / "results.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows))
    if not args.max_episodes_per_split:
        groups = {group: summary([row for row in rows if row["group"] == group])
                  for group in by_group}
        special = [row for row in rows if row["is_held_out_query"]]
        groups["held_out_composition_queries"] = summary(special)
        output = {"checkpoint": str(args.checkpoint), "checkpoint_step": saved["step"],
                  "protocol": "25 Hz, 30 query-to-current RGB-D frames, one decision and recovery mask frame, four valid 25-action chunks per episode, 8 Euler flow steps",
                  "groups": groups}
        (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps({"event": "complete", **output}), flush=True)


if __name__ == "__main__":
    main()
