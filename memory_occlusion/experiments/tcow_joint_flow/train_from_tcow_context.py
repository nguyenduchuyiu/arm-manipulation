"""Train TCOW masks on context and 25-step flow chunks every 10 action frames."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from memory_occlusion.experiments.tcow_joint_flow.data import (
    load_joint_episode, policy_statistics, training_clip, training_samples,
)
from memory_occlusion.experiments.tcow_joint_flow.model import JointTCOWFlow, expand_depth_channel
from memory_occlusion.experiments.tcow_joint_flow.data import split_rows
from memory_occlusion.experiments.tcow_joint_flow.tcow import (
    MyLosses, Seeker, checkpoint_transformer_blocks, original_mask_loss,
)
from memory_occlusion.experiments.tcow_joint_flow.validation import evaluate, evaluate_closed_loop


def load_cluster(root, rows):
    return [(row, load_joint_episode(root / row["path"]))
            for row in tqdm(rows, desc="load episodes", unit="episode", mininterval=5,
                            file=sys.stdout)]


def save_checkpoint(path, model, step, args, metrics):
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "step": step, "architecture": "TCOW RGB-D + dense 301-token context, flow 25x6",
               "flow_type": model.flow.flow_type,
               "source_flow_checkpoint": str(args.flow_weights) if args.flow_weights else None,
               "source_checkpoint": str(args.weights), "metrics": metrics,
               "mask_loss_weight": args.mask_loss_weight}
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--flow-weights", type=Path, required=True,
                   help="expert.safetensors exported by download_action_expert")
    p.add_argument("--config-checkpoint", type=Path,
                   help="required for joint TCOW + flow training")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cluster-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int,
                   help="default: 8 on CUDA, 1 on MPS")
    p.add_argument("--eval-every-clusters", type=int, default=20)
    p.add_argument("--max-clusters", type=int, default=0)
    p.add_argument("--max-val-episodes", type=int, default=80)
    p.add_argument("--closed-loop-every-evals", type=int, default=0,
                   help="run fixed validation scenes every N evaluations; 0 disables simulation")
    p.add_argument("--tcow-lr", type=float, default=2e-5)
    p.add_argument("--flow-lr", type=float, default=1e-4)
    p.add_argument("--mask-loss-weight", type=float, default=.2)
    p.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    p.add_argument("--flow-only", action="store_true",
                   help="freeze TCOW and optimize only the action flow")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = argument_parser().parse_args()
    if args.closed_loop_every_evals < 0:
        raise ValueError("closed-loop-every-evals must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if not args.flow_only and args.config_checkpoint is None:
        raise ValueError("--config-checkpoint is required for joint training")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if not args.flow_weights.is_file():
        raise FileNotFoundError(args.flow_weights)
    if args.config_checkpoint and not args.config_checkpoint.is_file():
        raise FileNotFoundError(args.config_checkpoint)
    device = args.device
    if args.batch_size is None:
        args.batch_size = 1 if device == "mps" else 8
    if min(args.cluster_size, args.epochs, args.batch_size) < 1:
        raise ValueError("cluster size, epochs, and batch size must be positive")
    train_rows, val_rows = split_rows(args.data)
    val_rows = val_rows[:args.max_val_episodes] if args.max_val_episodes else val_rows
    rng = np.random.default_rng(args.seed)
    train_rows = [train_rows[i] for i in rng.permutation(len(train_rows))]
    clusters = [train_rows[i:i + args.cluster_size]
                for i in range(0, len(train_rows), args.cluster_size)]
    if args.max_clusters:
        clusters = clusters[:args.max_clusters]
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sample_fps": 25, "context_frames": 30, "action_horizon": 25,
        "context_stride": 10, "action_stride": 10, "inference_execution_steps": 10,
        "flow_steps": 10, "flow_type": "dense_action_expert",
        "flow_time": "beta_sinusoidal_noise_at_t0",
        "train_episodes": sum(map(len, clusters)), "validation_episodes": len(val_rows),
    }, indent=2) + "\n")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    config = (torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
              if args.config_checkpoint else None)
    pretrained = torch.load(args.weights, map_location="cpu", weights_only=False, mmap=True)
    if pretrained.get("input_channels") != 5 or "model" in pretrained:
        raise ValueError("expected an exported TCOW-only RGB-D checkpoint")
    seeker_args = dict(config["seeker_args"] if config else pretrained["seeker_args"])
    seeker_args["tracker_pretrained"] = False
    tcow = expand_depth_channel(Seeker(logging.getLogger("tcow"), **seeker_args))
    tcow.load_state_dict(pretrained["net_seeker"], strict=True)
    if not args.flow_only:
        checkpoint_transformer_blocks(tcow)
    from memory_occlusion.experiments.tcow_joint_flow.flow_matching import DenseFlowMatching
    flow = DenseFlowMatching()
    imported = flow.load_pretrained(args.flow_weights)
    statistics = policy_statistics(args.data, [row for rows in clusters for row in rows])
    flow.set_statistics(statistics)
    (args.output / "flow_initialization.json").write_text(
        json.dumps({"import": imported, "normalization": statistics}, indent=2) + "\n")
    print(json.dumps({"event": "pretrained_flow", **imported}), flush=True)
    model = JointTCOWFlow(tcow, flow).to(device)
    if args.flow_only:
        model.tcow.requires_grad_(False)
    losses = (MyLosses(config["train_args"], logging.getLogger("tcow"), "train")
              if config else None)
    groups = [{"params": model.flow.parameters(), "lr": args.flow_lr,
               "betas": (.9, .95), "weight_decay": 1e-10}]
    if not args.flow_only:
        groups.insert(0, {"params": model.tcow.parameters(), "lr": args.tcow_lr})
    optimizer = torch.optim.AdamW(groups)
    # The exact step count is known after indexing action_valid, before video decoding.
    # Episodes are kept in small RAM clusters; every sample in each cluster is shuffled.
    from memory_occlusion.experiments.tcow_joint_flow.data import sample_index
    plans = [[[(end, has_action) for end, has_action in sample_index(args.data / row["path"])
               if has_action or not args.flow_only] for row in rows] for rows in
             tqdm(clusters, desc="index episodes", unit="cluster", file=sys.stdout)]
    total_steps = args.epochs * sum(
        sum((count + args.batch_size - 1) // args.batch_size for count in
            (sum(not has_action for plan in cluster for _, has_action in plan),
             sum(has_action for plan in cluster for _, has_action in plan)))
        for cluster in plans)
    total_samples = sum(len(plan) for cluster in plans for plan in cluster)
    total_action = sum(sum(has_action for _, has_action in plan)
                       for cluster in plans for plan in cluster)
    started = time.perf_counter()
    print(json.dumps({"event": "start", "steps": total_steps, "epochs": args.epochs,
                      "samples_per_epoch": total_samples,
                      "mask_only_per_epoch": total_samples - total_action,
                      "action_chunks_per_epoch": total_action,
                      "trainable_tcow": sum(p.numel() for p in model.tcow.parameters() if p.requires_grad),
                      "trainable_flow": sum(p.numel() for p in model.flow.parameters()),
                      "batch_size": args.batch_size, "device": device,
                      "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "Apple MPS",
                      "flow_only": args.flow_only,
                      "source_tcow_step": pretrained["source_step"],
                      "fresh_flow": False,
                      "flow_type": model.flow.flow_type}), flush=True)
    step = 0
    best = -1.0
    history = []
    wait_s = 0.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with ThreadPoolExecutor(max_workers=1) as loader:
        future = loader.submit(load_cluster, args.data, clusters[0])
        with tqdm(total=total_steps, desc="train TCOW + flow", unit="step", mininterval=5,
                  file=sys.stdout) as progress:
            for cluster_index in range(len(clusters) * args.epochs):
                cluster_id = cluster_index % len(clusters)
                waiting = time.perf_counter()
                episodes = future.result()
                wait_s += time.perf_counter() - waiting
                next_id = (cluster_index + 1) % len(clusters)
                future = (loader.submit(load_cluster, args.data, clusters[next_id])
                          if cluster_index + 1 < len(clusters) * args.epochs else None)
                items = [(episode_id, end, has_action)
                         for episode_id, plan in enumerate(plans[cluster_id])
                         for end, has_action in plan]
                for episode_id, (_row, episode) in enumerate(episodes):
                    indexed = [(end, has_action) for end, has_action in training_samples(episode)
                               if has_action or not args.flow_only]
                    if indexed != plans[cluster_id][episode_id]:
                        raise ValueError(f"indexed samples changed: {episode['name']}")
                batches = []
                for action_phase in (False, True):
                    phase_items = [item for item in items if item[2] == action_phase]
                    rng.shuffle(phase_items)
                    batches.extend(phase_items[i:i + args.batch_size]
                                   for i in range(0, len(phase_items), args.batch_size))
                rng.shuffle(batches)
                for batch_items in batches:
                    samples = [training_clip(episodes[episode_id][1], end, has_action, device)
                               for episode_id, end, has_action in batch_items]
                    rgbd, query, truth, proprio, action, valid_steps = (
                        torch.cat(parts, dim=0) for parts in zip(*samples))
                    has_action = valid_steps.any(dim=1)
                    model.train()
                    if args.flow_only:
                        model.tcow.eval()
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        with torch.set_grad_enabled(not args.flow_only):
                            logits, _ = model.tcow(rgbd, query)
                        latent = model._latent
                        if has_action.any():
                            noisy, tau, target_velocity = model.flow.training_path(action[has_action])
                            velocity, context = model.flow(latent[has_action], proprio[has_action], noisy, tau)
                            squared = (velocity.float() - target_velocity).square()
                            weights = valid_steps[has_action, :, None]
                            action_loss = (squared * weights).sum() / (weights.sum() * action.shape[-1])
                        else:
                            action_loss = logits.new_zeros(())
                    mask_loss = (logits.new_zeros(()) if args.flow_only else
                                 original_mask_loss(losses, logits, truth, step / max(total_steps, 1)))
                    total = action_loss + args.mask_loss_weight * mask_loss
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    model._latent = None
                    if step == 0:
                        patch_grad = model.tcow.seeker.tracker_backbone.timesformer.model.patch_embed.proj.weight.grad
                        head_grad = model.tcow.seeker.tracker_post_linear.weight.grad
                        flow_grad = model.flow.output.weight.grad
                        print(json.dumps({"event": "gradient_smoke", "latent_shape": list(latent.shape),
                                          "context_shape": list(context.shape) if has_action.any() else None,
                                          "patch_grad": float(patch_grad.float().abs().sum()) if patch_grad is not None else None,
                                          "mask_head_grad": float(head_grad.float().abs().sum()) if head_grad is not None else None,
                                          "flow_grad": float(flow_grad.float().abs().sum()) if flow_grad is not None else None,
                                          "loss_action": float(action_loss), "loss_mask": float(mask_loss),
                                          "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else None}), flush=True)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config["train_args"].gradient_clip if config else 1.0)
                    optimizer.step()
                    step += 1
                    progress.update(1)
                    progress.set_postfix(action=f"{action_loss.item():.3f}",
                                         mask=f"{mask_loss.item():.3f}",
                                         chunks=int(has_action.sum()), refresh=False)
                del episodes
                if ((cluster_index + 1) % args.eval_every_clusters == 0 or
                        cluster_index + 1 == len(clusters) * args.epochs):
                    metrics = evaluate(model, args.data, val_rows, device)
                    if args.closed_loop_every_evals and (len(history) + 1) % args.closed_loop_every_evals == 0:
                        metrics["closed_loop"] = evaluate_closed_loop(
                            model, args.data, val_rows, args.output / "validation" / f"step_{step}", step)
                    value = metrics["action_mae"]
                    if best < 0 or value < best:
                        best = value
                        save_checkpoint(args.output / "best.pt", model, step, args, metrics)
                    record = {"step": step, "clusters_done": cluster_index + 1,
                              "epoch": cluster_index // len(clusters) + 1,
                              "action_loss": float(action_loss), "mask_loss": float(mask_loss),
                              "validation": metrics, "best_action_mae": best,
                              "elapsed_s": time.perf_counter() - started,
                              "prefetch_wait_s": wait_s,
                              "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else None,
                              "peak_cpu_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
                    history.append(record)
                    tqdm.write(json.dumps(record), file=sys.stdout)
    save_checkpoint(args.output / "final.pt", model, step, args, history[-1]["validation"])
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps({"event": "complete", "steps": step,
                      "elapsed_s": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
