"""Finetune TCOW masks on complete context+expert episodes before freezing it."""
import argparse
import json
import logging
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from OATFlow.policy.data import split_rows, tracking_starts
from OATFlow.policy.loader import (ActionChunkDataset, cache_videos, make_dataloader,
                                  release_videos, split_training_batches, training_batches)
from OATFlow.policy.model import load_training_tracker, set_tracker_input
from OATFlow.policy.tracking import (MyLosses, Seeker, checkpoint_transformer_blocks,
                                    original_mask_loss)
from OATFlow.policy.evaluate_tcow import evaluate, high_iou, validation_plan


def balanced_training_subset(rows, per_task, seed):
    """Sample train scenes only, with equal task and target identity quotas."""
    if per_task is None:
        return rows
    if per_task < 2 or per_task % 2:
        raise ValueError("subset demos per task must be a positive even number")
    rng = np.random.default_rng(seed)
    selected = []
    for task in range(12):
        task_rows = [row for row in rows if row["task_id"] == task]
        targets = sorted({row["target_id"] for row in task_rows})
        if len(targets) != 2:
            raise ValueError(f"task{task} must contain two target identities")
        for target in targets:
            candidates = [row for row in task_rows if row["target_id"] == target]
            if any(row["split"] != "train" or row["group"] != "standard" for row in candidates):
                raise ValueError("TCOW subset must use standard/train scenes only")
            count = per_task // 2
            if len(candidates) < count:
                raise ValueError(f"insufficient train demos for task{task}/target{target}")
            selected.extend(candidates[index] for index in rng.permutation(len(candidates))[:count])
    if len({row["seed"] for row in selected}) != len(selected):
        raise ValueError("TCOW subset scene seeds must be unique")
    return selected


def save_tracker(path, model, step, epoch, seeker_args, train_args, actual,
                 validation, status, optimizer=None, scheduler=None):
    payload = {"net_seeker": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "step": step, "source_step": step, "epoch": epoch,
               "visual_input": "rgb", "tcow_input_channels": 4,
               "seeker_args": seeker_args, "train_args": train_args, "config": actual,
               "validation": validation, "training_status": status}
    if optimizer is not None:
        payload.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state())
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "weights", "config-checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--cluster-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--warmup-steps", type=int, default=215)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--early-stop-iou", type=float, default=.85)
    parser.add_argument("--subset-per-task", type=int, help="balanced standard/train scenes per task; half per target")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if min(args.epochs, args.batch_size, args.cluster_size) < 1:
        raise ValueError("positive epochs, batch and cluster size required")
    if not 1 <= args.micro_batch_size <= args.batch_size:
        raise ValueError("micro batch must be 1..batch size")
    if not 0 < args.min_lr <= args.lr or args.warmup_steps < 1:
        raise ValueError("positive LR floor <= peak and positive warmup required")
    if args.eval_every < 1 or not 0 < args.early_stop_iou <= 1:
        raise ValueError("positive evaluation interval and IoU threshold in(0,1] required")
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.weights, args.config_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    rows, val_rows = split_rows(args.data)
    available_train = len(rows)
    rows = balanced_training_subset(rows, args.subset_per_task, args.seed)
    if not val_rows:
        raise ValueError("scene-disjoint validation episodes required for early stopping")
    rng = np.random.default_rng(args.seed)
    rows = [rows[index] for index in rng.permutation(len(rows))]
    plans = [[[(end, False) for end in tracking_starts(row["frames"])]
              for row in rows[start:start + args.cluster_size]]
             for start in range(0, len(rows), args.cluster_size)]
    samples, batches, epoch_ends = training_batches(plans, args.epochs, args.batch_size, rng)
    physical, weights = split_training_batches(batches, samples, rows, args.micro_batch_size)
    if args.warmup_steps >= len(batches):
        raise ValueError("warmup must end before final training step")
    config = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    source = torch.load(args.weights, map_location="cpu", weights_only=False, mmap=True)
    seeker_args = dict(config["seeker_args"])
    seeker_args["tracker_pretrained"] = False
    model = set_tracker_input(Seeker(logging.getLogger("tcow"), **seeker_args))
    source_step = load_training_tracker(model, source)
    checkpoint_transformer_blocks(model)
    model = model.cuda().train()
    losses = MyLosses(config["train_args"], logging.getLogger("tcow"), "train")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=1e-10)
    scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=args.warmup_steps, num_decay_steps=len(batches),
        peak_lr=args.lr, decay_lr=args.min_lr).build(optimizer, num_training_steps=len(batches))
    args.output.mkdir(parents=True)
    actual = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "stage": "tcow_mask_finetune", "source_step": source_step,
              "train_episodes": len(rows), "samples_per_epoch": len(samples),
              "available_train_episodes": available_train,
              "optimizer_steps": len(batches), "physical_batches": len(physical),
              "optimizer_epoch_ends": epoch_ends, "context_frames": 30,
              "visual_input": "rgb", "tcow_input_channels": 4, "mask_channels": 3,
              "stride": 1,
              "lr_scheduler": "cosine_decay_with_warmup", "fm_training": False,
              "validation": "all standard/val scenes; visible/shuffle/post-shuffle/expert endpoints",
              "validation_scenes": len(val_rows), "checkpoint_every": args.eval_every,
              "early_stop_rule": "all3 foreground IoUs, hidden target and post-shuffle covers >= threshold; absent FP pixel rate<=0.005"}
    (args.output / "config.json").write_text(json.dumps(actual, indent=2) + "\n")
    (args.output / "train_subset.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "README.md").write_text(
        f"# {args.output.name}\n\nTCOW-only mask finetuning; no FM/wrist modules or action loss.\n"
        f"Data {args.data}; {len(rows)} standard/train episodes, complete context+expert. "
        f"Query first-frame target mask; 30 past/current RGB frames, crop240x320, three mask channels.\n"
        f"Source {args.weights}, config {args.config_checkpoint}, source step {source_step}.\n"
        f"Epochs {args.epochs}; batch {args.batch_size}, micro {args.micro_batch_size}; stride1. "
        f"Warmup{args.warmup_steps}, cosine {args.lr}→{args.min_lr}.\n"
        "TCOW backbone/mask head trainable. Latest/final tracker initializes frozen TCOW in the FM stage. "
        f"Save/evaluate every{args.eval_every}updates and at epoch ends, plus baseline evaluation. "
        f"Early stop threshold{args.early_stop_iou} on foreground/hidden-target/post-shuffle-cover IoU; "
        "empty GT scored separately as false positive area. Final selects best validation snapshot. "
        "Tests: no action rollout run. Exact config: config.json. Status: training.\n")
    videos, gib = cache_videos(args.data, rows + val_rows, False)
    (args.output / "video_cache.json").write_text(json.dumps(
        {"directory": str(Path(next(iter(videos.values()))).parent), "gib": gib}, indent=2) + "\n")
    loader = make_dataloader(ActionChunkDataset(args.data, rows, samples, videos, False, False),
                             physical, args.seed, True)
    val_samples, val_groups = validation_plan(args.data, val_rows)
    val_batches = [list(range(start, min(start + args.micro_batch_size, len(val_samples))))
                   for start in range(0, len(val_samples), args.micro_batch_size)]
    val_loader = make_dataloader(ActionChunkDataset(args.data, val_rows, val_samples, videos, False, False),
                                 val_batches, args.seed, True)
    history, step, checked = [], 0, False
    best_score, requested, stopped_early = -1., False, False
    def request_eval(_signal, _frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGUSR1, request_eval)
    def validate_checkpoint():
        nonlocal best_score, requested
        optimizer.zero_grad(set_to_none=True)
        result = evaluate(model, val_loader, val_samples, val_groups, val_rows)
        model.train()
        epoch = sum(step >= end for end in epoch_ends)
        metrics = {k: v for k, v in result.items() if k != "records"}
        ready = high_iou(result, args.early_stop_iou)
        score = result["mean_foreground_iou"]
        if score is None:
            raise ValueError("validation must contain positive masks in all3channels")
        result.update(step=step, epoch=epoch, early_stop_ready=ready, threshold=args.early_stop_iou)
        evaluation_dir = args.output / "validation"
        evaluation_dir.mkdir(exist_ok=True)
        (evaluation_dir / f"step_{step:06d}.json").write_text(json.dumps(result, indent=2) + "\n")
        if score is not None and score > best_score:
            best_score = score
            save_tracker(args.output / "best.pt", model, step, epoch, seeker_args, config["train_args"],
                         actual, result, "training")
        # A qualifying snapshot is selected immediately, even if a previous
        # snapshot had a higher aggregate but failed a critical hidden-case metric.
        if ready:
            save_tracker(args.output / "best.pt", model, step, epoch, seeker_args, config["train_args"],
                         actual, result, "early_stopped")
        save_tracker(args.output / "latest.pt", model, step, epoch, seeker_args, config["train_args"],
                     actual, result, "early_stopped" if ready else "training", optimizer, scheduler)
        record = {"step": step, "epoch": epoch, "early_stop_ready": ready, **metrics}
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"event": "validation", **record}), flush=True)
        requested = False
        return ready
    started = last_refresh = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"event": "start", **actual}), flush=True)
    iterator = iter(loader)
    try:
        stopped_early = validate_checkpoint()
        with tqdm(total=len(batches), desc="finetune TCOW", unit="step", mininterval=5,
                  file=sys.stdout) as progress:
            for logical_size, _, first, last, consumed in weights:
                if stopped_early:
                    break
                if first:
                    optimizer.zero_grad(set_to_none=True)
                    logical_loss = 0.
                tensors = next(iterator)
                rgb, query, truth = [tensor.cuda(non_blocking=True) for tensor in tensors[:3]]
                del tensors
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(rgb, query)
                loss = original_mask_loss(losses, logits, truth, step / len(batches))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite TCOW mask loss")
                scale = len(rgb) / logical_size
                (loss * scale).backward()
                logical_loss += float(loss.detach()) * scale
                if not checked:
                    patch = model.seeker.tracker_backbone.timesformer.model.patch_embed.proj.weight.grad
                    head = model.seeker.tracker_post_linear.weight.grad
                    if any(g is None or not torch.isfinite(g).all() or not g.abs().any() for g in (patch, head)):
                        raise ValueError("missing or nonfinite TCOW gradient")
                    print(json.dumps({"event": "gradients_checked", "patch_grad": float(patch.abs().sum()),
                                      "mask_head_grad": float(head.abs().sum()), "loss_mask": float(loss.detach()),
                                      "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30}), flush=True)
                    checked = True
                if last:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["train_args"].gradient_clip,
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    progress.update(1)
                progress.set_postfix(mask=f"{logical_loss:.3f}", accumulated=f"{consumed}/{logical_size}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.2e}", refresh=False)
                if time.perf_counter() - last_refresh > 10:
                    progress.refresh()
                    last_refresh = time.perf_counter()
                del rgb, query, truth, logits, loss
                if last and (step % args.eval_every == 0 or step in epoch_ends or requested):
                    stopped_early = validate_checkpoint()
    finally:
        del iterator, loader, val_loader
        release_videos(videos)
    selected = torch.load(args.output / "best.pt", map_location="cpu", weights_only=False, mmap=True)
    selected["training_status"] = "early_stopped" if stopped_early else "complete"
    selected["completed_training_steps"] = step
    temporary = args.output / "final.tmp"
    torch.save(selected, temporary)
    temporary.replace(args.output / "final.pt")
    p = args.output / "README.md"
    p.write_text(p.read_text().replace("Status: training.", "Status: complete."))
    print(json.dumps({"event": "complete", "step": step, "selected_step": selected["step"],
                      "early_stopped": stopped_early, "validation": selected["validation"]["mean_foreground_iou"]}), flush=True)


if __name__ == "__main__":
    main()
