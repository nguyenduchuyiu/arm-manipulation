"""Validation on every training-aligned action chunk and fixed held-out rollouts."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from OATFlow.experiments.tcow_joint_flow.data import (
    load_joint_episode, training_clip, training_samples,
)


@torch.inference_mode()
def evaluate(model, root, rows, device, distributed=False):
    if not rows or any(row["split"] != "val" for row in rows):
        raise ValueError("validation requires nonempty validation-only rows")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    model.eval()
    # Rows: absolute error sum for six joints; valid counts per chunk offset.
    errors = np.zeros((25, 6), np.float64)
    counts = np.zeros(25, np.float64)
    # Per close phase: gripper absolute error, binary error, valid count.
    closing = np.zeros((2, 3), np.float64)
    ious = np.zeros(3, np.float64)
    chunks = 0
    use_wrist = getattr(model.flow, "wrist_enabled", False)
    for row in tqdm(rows[rank::world], desc="validate episodes", unit="episode",
                    mininterval=5, file=sys.stdout, disable=rank != 0):
        path = root / row["path"]
        episode = load_joint_episode(path, require_wrist=use_wrist)
        meta = json.loads((path / "episode.json").read_text())
        phases = np.zeros((2, len(episode["action_valid"])), dtype=bool)
        for index, name in enumerate(("cover_close", "object_close")):
            intervals = [p for p in meta["phase_intervals"] if p["phase"] == name]
            if not intervals:
                raise ValueError(f"missing {name}: {path}")
            for interval in intervals:
                phases[index, interval["start_frame"]:interval["end_frame_exclusive"]] = True
        ends = [end for end, has_action in training_samples(episode) if has_action]
        for end in tqdm(ends, desc=f"validate {path.name}", unit="chunk", leave=False,
                        mininterval=5, file=sys.stdout, disable=rank != 0):
            tensors = training_clip(episode, end, True, device, include_wrist=use_wrist)
            rgbd, query, truth, proprio, action, valid = tensors[:6]
            wrist = tensors[6] if use_wrist else None
            if episode.get("action_representation") == "relative_joint":
                action = action.clone()
                action[:, :, :5] += proprio[:, None, :5]
            generator_device = "cpu" if device == "mps" else device
            generator = torch.Generator(device=generator_device).manual_seed(int(row["seed"]) + end)
            noise = model.flow.sample_noise(1, generator_device, generator).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                model._latent = None
                logits, _ = model.tcow(rgbd, query)
                estimate = (model.flow.sample(model._latent, proprio, noise, wrist=wrist) if use_wrist else
                            model.flow.sample(model._latent, proprio, noise))
            if not torch.isfinite(estimate).all():
                raise ValueError(f"nonfinite validation action: {path}, frame {end}")
            predicted = estimate[0].float().cpu().numpy()
            target_action = action[0].float().cpu().numpy()
            mask = valid[0].cpu().numpy()
            error = np.abs(predicted - target_action)
            errors += error * mask[:, None]
            counts += mask
            length = min(25, phases.shape[1] - end)
            for index in range(2):
                keep = mask[:length] & phases[index, end:end + length]
                closing[index] += [error[:length, 5][keep].sum(),
                                   ((predicted[:length, 5] >= .5) !=
                                    (target_action[:length, 5] >= .5))[keep].sum(), keep.sum()]
            prediction = logits[0, :, -1].float() > 0
            target_mask = truth[0, :, -1].bool()
            ious += ((prediction & target_mask).sum(dim=(-1, -2)).float() /
                     (prediction | target_mask).sum(dim=(-1, -2)).clamp(min=1)).cpu().numpy()
            chunks += 1
        del episode
    if distributed:
        packed = torch.tensor(np.r_[errors.ravel(), counts, closing.ravel(), ious, chunks],
                              dtype=torch.float64, device=device)
        dist.all_reduce(packed)
        values = packed.cpu().numpy()
        errors, counts = values[:150].reshape(25, 6), values[150:175]
        closing, ious, chunks = values[175:181].reshape(2, 3), values[181:184], int(values[184])
    if not chunks or not counts.all():
        raise ValueError("validation has no complete action chunks")
    result = {"protocol": "all_valid_starts_stride10", "action_units": "joint_limits",
              "episodes": len(rows), "action_chunks": chunks, "action_stride": 10,
              "action_valid_steps": int(counts.sum()),
              "action_mae": float(errors.sum() / (counts.sum() * 6)),
              "action_mae_first": float(errors[0].sum() / (counts[0] * 6)),
              "action_mae_first10": float(errors[:10].sum() / (counts[:10].sum() * 6)),
              "action_mae_per_offset": (errors.sum(axis=1) / (counts * 6)).tolist(),
              "action_mae_per_joint": (errors.sum(axis=0) / counts.sum()).tolist(),
              **dict(zip(("target_iou", "occluder_iou", "container_iou"), (ious / chunks).tolist()))}
    for name, (absolute, wrong, count) in zip(("cover_close", "object_close"), closing):
        result[name] = {"gripper_mae": float(absolute / count) if count else None,
                        "gripper_error_rate": float(wrong / count) if count else None,
                        "valid_steps": int(count)}
    return result


def closed_loop_rows(rows):
    if not rows or any(row["split"] != "val" for row in rows):
        raise ValueError("closed-loop validation requires validation-only rows")
    selected, seeds = [], set()
    for target in sorted({row["target"] for row in rows}):
        candidates = sorted((row for row in rows if row["target"] == target),
                            key=lambda row: (row["seed"], row["path"]))
        row = next((row for row in candidates if row["seed"] not in seeds), candidates[0])
        selected.append(row)
        seeds.add(row["seed"])
    return selected


def evaluate_closed_loop(model, root, rows, output, step):
    from OATFlow.experiments.tcow_joint_flow.rollout import run_rollout
    selected = closed_loop_rows(rows)
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(selected, indent=2) + "\n")
    results = []
    for row in tqdm(selected, desc="validate closed loop", unit="episode", file=sys.stdout):
        args = SimpleNamespace(reference_episode=root / row["path"],
                               output=output / Path(row["path"]).name,
                               max_total_frames=1000, execute_chunk=10)
        results.append(run_rollout(model, args, step))
    summary = {"episodes": len(results), "execute_chunk": 10, "max_total_frames": 1000,
               **{key + "_rate": float(np.mean([r[key] for r in results]))
                  for key in ("cover_choice_correct", "cover_removed", "target_grasped", "success")}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
