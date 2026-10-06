"""Train OAT-Flow with H25 action chunks and optional TCOW mask supervision."""
from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import math
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from OATFlow.policy.data import (
    expand_demonstrations, policy_statistics, require_wrist_data, sample_index, split_rows,
)
from OATFlow.policy.model import (
    FlowMatchingHead, OATFlowPolicy, checkpoint_uses_depth, load_training_tracker, set_tracker_input,
)
from OATFlow.policy.loader import (
    ActionChunkDataset, cache_videos, make_dataloader, release_videos, training_batches,
)
from OATFlow.policy.tracking import (
    MyLosses, Seeker, checkpoint_transformer_blocks, original_mask_loss,
)


def save_checkpoint(path, model, step, args):
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "step": step, "visual_input": "rgb", "tcow_input_channels": 4,
               "architecture": f"TCOW RGB + query mask + wrist/proprio, {model.flow.context_tokens}-token "
                               f"{'contextualized' if model.flow.contextualize else 'dense'} context, flow 25x{model.flow.action_dim}",
               "wrist_camera": model.flow.wrist_enabled,
               "wrist_imagenet_normalization": model.flow.wrist_enabled and model.flow.wrist_encoder.imagenet_normalization,
               "wrist_encoder_frozen": model.flow.wrist_enabled and model.flow.wrist_encoder.frozen,
               "source_wrist_checkpoint": str(args.wrist_weights) if args.wrist_weights else None,
               "context_tokens": model.flow.context_tokens,
               "wrist_tokens": model.flow.wrist_tokens,
               "contextualize": model.flow.contextualize,
               "context_decoder_layers": 4 if model.flow.contextualize else 0,
               "context_initialization": "random" if model.flow.contextualize else None,
               "tcow_frozen": not any(p.requires_grad for p in model.tcow.parameters()),
               "flow_type": model.flow.flow_type,
               "action_representation": model.flow.action_representation,
               "action_dim": model.flow.action_dim,
               "source_flow_checkpoint": str(args.flow_weights) if args.flow_weights else None,
               "source_checkpoint": str(args.weights), "metrics": {},
               "validation_enabled": False, "checkpoint_selection": "latest_epoch",
               "mask_loss_weight": args.mask_loss_weight}
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def argument_parser(distributed=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--base-demonstrations-only", action="store_true",
                   help="train expert parent episodes without attached perturbation demonstrations")
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--flow-weights", type=Path, required=True,
                   help="prepared pretrained action expert.safetensors")
    p.add_argument("--config-checkpoint", type=Path,
                   help="required for joint TCOW + flow training")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cluster-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int,
                   help="default: 8 on CUDA, 1 on MPS")
    p.add_argument("--max-clusters", type=int, default=0)
    p.add_argument("--smoke-steps", type=int, choices=(0, 4, 8) if distributed else (0, 4), default=0,
                   help="verify mask and action gradients; DDP eight-step mode crosses a cluster boundary")
    p.add_argument("--tcow-lr", type=float, default=2e-5)
    p.add_argument("--flow-lr", type=float, default=1e-4)
    p.add_argument("--wrist-lr", type=float, help="wrist backbone LR; defaults to flow LR")
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--warmup-fraction", type=float, default=.05)
    p.add_argument("--min-lr-ratio", type=float, default=.1)
    p.add_argument("--mask-loss-weight", type=float, default=.2)
    p.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    p.add_argument("--flow-only", action="store_true",
                   help="freeze TCOW and optimize only the action flow")
    p.add_argument("--wrist-camera", action=argparse.BooleanOptionalAction, default=True,
                   help="default: use current wrist RGB; --no-wrist-camera reproduces overview-only training")
    p.add_argument("--wrist-weights", type=Path,
                   help="torchvision ViT-B/16 ImageNet-1K V1 state dict for wrist initialization")
    p.add_argument("--freeze-wrist-encoder", action="store_true",
                   help="freeze the wrist ViT; wrist projection and FM remain trainable")
    p.add_argument("--contextualize", action="store_true",
                   help="pool wrist to 64 tokens and train a four-layer context decoder from scratch")
    p.add_argument("--seed", type=int, default=0)
    return p


def make_optimizer(model, args):
    wrist = list(model.flow.wrist_encoder.parameters()) if model.flow.wrist_enabled else []
    wrist_ids = {id(p) for p in wrist}
    groups = [{"name": "flow", "params": [p for p in model.flow.parameters()
               if p.requires_grad and id(p) not in wrist_ids], "lr": args.flow_lr,
               "betas": (.9, .95), "weight_decay": 1e-10}]
    if any(p.requires_grad for p in wrist):
        groups.append({"name": "wrist", "params": [p for p in wrist if p.requires_grad],
                       "lr": args.wrist_lr if args.wrist_lr is not None else args.flow_lr,
                       "betas": (.9, .95), "weight_decay": 1e-10})
    if not args.flow_only:
        groups.insert(0, {"name": "tcow", "params": [p for p in model.tcow.parameters()
                          if p.requires_grad], "lr": args.tcow_lr})
    if any(group["lr"] <= 0 for group in groups):
        raise ValueError("learning rates must be positive")
    return torch.optim.AdamW(groups)


def make_scheduler(optimizer, args, total_steps):
    if total_steps < 1 or not 0 <= args.warmup_fraction < 1 or not 0 < args.min_lr_ratio <= 1:
        raise ValueError("invalid step count, warmup fraction or minimum LR ratio")
    warmup = max(1, int(total_steps * args.warmup_fraction)) if args.warmup_fraction else 0

    def factor(index):
        if args.lr_schedule == "constant":
            return 1.
        if index < warmup:
            return (index + 1) / warmup
        progress = min(1., max(0., (index - warmup) / max(1, total_steps - warmup - 1)))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * .5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    faulthandler.dump_traceback_later(120, repeat=True)
    args = argument_parser().parse_args()
    if (args.wrist_weights or args.freeze_wrist_encoder) and not args.wrist_camera:
        raise ValueError("wrist initialization/freezing requires --wrist-camera")
    if args.wrist_weights and not args.wrist_weights.is_file():
        raise FileNotFoundError(args.wrist_weights)
    if args.contextualize:
        if not (args.wrist_camera and args.wrist_weights):
            raise ValueError("contextualization requires a pretrained wrist ViT")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if not args.flow_only and args.config_checkpoint is None:
        raise ValueError("--config-checkpoint is required for joint training")
    if args.smoke_steps and args.flow_only:
        raise ValueError("smoke steps require joint mask and action training")
    if args.smoke_steps:
        args.max_clusters = 1
        args.epochs = 1
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    if not args.flow_weights.is_file():
        raise FileNotFoundError(args.flow_weights)
    if args.config_checkpoint and not args.config_checkpoint.is_file():
        raise FileNotFoundError(args.config_checkpoint)
    device = args.device
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    if args.batch_size is None:
        args.batch_size = 1 if device == "mps" else 8
    if min(args.cluster_size, args.epochs, args.batch_size) < 1:
        raise ValueError("cluster size, epochs, and batch size must be positive")
    train_rows, _ = split_rows(args.data)
    train_parents = len(train_rows)
    train_rows = ([{**row, "parent_path": row["path"], "demonstration_id": "base"} for row in train_rows]
                  if args.base_demonstrations_only else expand_demonstrations(args.data, train_rows))
    if args.wrist_camera:
        require_wrist_data(args.data, train_rows)
    rng = np.random.default_rng(args.seed)
    train_rows = [train_rows[i] for i in rng.permutation(len(train_rows))]
    clusters = [train_rows[i:i + args.cluster_size]
                for i in range(0, len(train_rows), args.cluster_size)]
    if args.max_clusters:
        clusters = clusters[:args.max_clusters]
    if args.output.exists():
        raise FileExistsError(args.output)
    statistics = policy_statistics(args.data, [row for rows in clusters for row in rows])
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sample_fps": 25, "context_frames": 30, "action_horizon": 25,
        "context_stride": 10, "action_stride": 10, "inference_execution_steps": 10,
        "flow_steps": 10, "flow_type": "dense_action_expert",
        "flow_time": "beta_sinusoidal_noise_at_t0",
        "train_episodes": sum(map(len, clusters)),
        "train_parent_episodes": len({row["parent_path"] for rows in clusters for row in rows}),
        "available_train_parent_episodes": train_parents,
        "validation_episodes": 0,
        "validation_enabled": False,
        "visual_input": "rgb", "tcow_input_channels": 4,
        "contextualize": args.contextualize,
        "context_tokens": 365 if args.contextualize else (701 if args.wrist_camera else 301),
        "action_representation": statistics.get("action_representation", "absolute_joint"),
        "action_dim": len(statistics["action_mean"]),
        "video_loading": "lerobot_torchcodec_ram",
        "video_decoder_cache_size": 8,
        "dataloader": "torch.utils.data.DataLoader",
        "num_workers": 2, "prefetch_factor": 2, "persistent_workers": True,
        "pin_memory": device == "cuda", "multiprocessing_context": "spawn",
    }, indent=2) + "\n")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    config = (torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
              if args.config_checkpoint else None)
    pretrained = torch.load(args.weights, map_location="cpu", weights_only=False, mmap=True)
    seeker_args = dict(config["seeker_args"] if config else pretrained["seeker_args"])
    seeker_args["tracker_pretrained"] = False
    tcow = set_tracker_input(Seeker(logging.getLogger("tcow"), **seeker_args))
    source_step = load_training_tracker(tcow, pretrained, args.flow_only)
    if not args.flow_only:
        checkpoint_transformer_blocks(tcow)
    flow = FlowMatchingHead(wrist_backbone=tcow.seeker.tracker_backbone if args.wrist_camera else None,
                            contextualize=args.contextualize)
    imported = flow.load_pretrained(args.flow_weights)
    if args.contextualize:
        imported["context_initialization"] = {"initialization": "random", "layers": 4, "width": 960,
                                              "parameters": sum(p.numel() for p in flow.context_decoder.parameters())}
    if args.wrist_weights:
        imported["wrist_initialization"] = flow.wrist_encoder.load_vit_base(args.wrist_weights)
    if args.freeze_wrist_encoder:
        flow.wrist_encoder.freeze()
    imported["wrist_encoder_frozen"] = args.freeze_wrist_encoder
    flow.relative_actions = statistics.get("action_representation") == "relative_joint"
    flow.set_statistics(statistics)
    if "model" in pretrained:
        flow.visual.load_state_dict({k.removeprefix("flow.visual."): v for k, v in pretrained["model"].items()
                                     if k.startswith("flow.visual.")}, strict=True)
        imported["visual_initialization"] = str(args.weights)
    imported["action_representation"] = flow.action_representation
    imported["tcow_input"] = {"visual_input": "rgb", "patch_channels": 4,
                              "source_patch_channels": 5 if checkpoint_uses_depth(pretrained) else 4,
                              "conversion": "keep RGB and query kernels; discard depth kernel"}
    (args.output / "flow_initialization.json").write_text(
        json.dumps({"import": imported, "normalization": statistics}, indent=2) + "\n")
    print(json.dumps({"event": "pretrained_flow", **imported}), flush=True)
    model = OATFlowPolicy(tcow, flow).to(device)
    if args.flow_only:
        model.tcow.requires_grad_(False)
    losses = (MyLosses(config["train_args"], logging.getLogger("tcow"), "train")
              if config else None)
    optimizer = make_optimizer(model, args)
    # The exact step count is known after indexing action_valid, before video decoding.
    # Keep the original cluster/sample order; only image loading changes.
    plans = [[[(end, has_action) for end, has_action in sample_index(args.data / row["path"])
               if has_action or not args.flow_only] for row in rows] for rows in
             tqdm(clusters, desc="index episodes", unit="cluster", file=sys.stdout)]
    samples, batches, epoch_ends = training_batches(
        plans, args.epochs, args.batch_size, rng, args.smoke_steps)
    total_steps = len(batches)
    scheduler = make_scheduler(optimizer, args, total_steps)
    total_samples = len(samples)
    total_action = sum(active for _episode, _end, active in samples)
    videos, compressed_gib = cache_videos(args.data, [row for rows in clusters for row in rows], args.wrist_camera)
    print(json.dumps({"event": "video_cache", "videos": len(videos),
                      "compressed_gib": compressed_gib, "video_decoder_cache_size": 8,
                      "backend": "lerobot/torchcodec",
                      "num_workers": 2, "prefetch_factor": 2,
                      "prefetched_batches": 4}), flush=True)
    cache_directory = str(Path(next(iter(videos.values()))).parent)
    (args.output / "video_cache.json").write_text(json.dumps({
        "directory": cache_directory, "compressed_gib": compressed_gib,
        "shared_with_workers": True, "filesystem": "tmpfs"}, indent=2) + "\n")
    dataset = ActionChunkDataset(args.data, [row for rows in clusters for row in rows],
                            samples, videos, args.flow_only, args.wrist_camera)
    dataloader = make_dataloader(dataset, batches, args.seed, device == "cuda")
    started = time.perf_counter()
    print(json.dumps({"event": "start", "steps": total_steps, "epochs": args.epochs,
                      "samples_per_epoch": total_samples,
                      "mask_only_per_epoch": total_samples - total_action,
                      "action_chunks_per_epoch": total_action,
                      "trainable_tcow": sum(p.numel() for p in model.tcow.parameters() if p.requires_grad),
                      "trainable_flow": sum(p.numel() for p in model.flow.parameters() if p.requires_grad),
                      "trainable_wrist_encoder": sum(p.numel() for p in model.flow.wrist_encoder.parameters()
                                                     if p.requires_grad) if args.wrist_camera else 0,
                      "batch_size": args.batch_size, "device": device,
                      "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "Apple MPS",
                      "flow_only": args.flow_only,
                      "source_tcow_step": source_step,
                      "fresh_flow": False,
                      "flow_type": model.flow.flow_type}), flush=True)
    step = 0
    history = []
    wait_s = 0.0
    checked_phases = set()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    iterator = iter(dataloader)
    with tqdm(total=total_steps, desc="train TCOW + flow", unit="step", mininterval=5,
              file=sys.stdout) as progress:
        for batch_index in range(total_steps):
            waiting = time.perf_counter()
            tensors = next(iterator)
            wait_s += time.perf_counter() - waiting
            step_started = time.perf_counter()
            tensors = [tensor.to(device, non_blocking=device == "cuda") for tensor in tensors]
            rgbd, query, truth, proprio, action, valid_steps = tensors[:6]
            wrist = tensors[6] if args.wrist_camera else None
            del tensors
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
                    velocity, context = model.flow(
                        latent[has_action], proprio[has_action], noisy, tau,
                        wrist=wrist[has_action] if wrist is not None else None)
                    squared = (velocity.float() - target_velocity).square()
                    weights = valid_steps[has_action, :, None]
                    action_loss = (squared * weights).sum() / (weights.sum() * action.shape[-1])
                else:
                    action_loss = logits.new_zeros(())
            mask_loss = (logits.new_zeros(()) if args.flow_only else
                         original_mask_loss(losses, logits, truth, step / max(total_steps, 1)))
            total = action_loss + args.mask_loss_weight * mask_loss
            if not torch.isfinite(total):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            model._latent = None
            action_phase = bool(has_action.any())
            if action_phase not in checked_phases:
                patch_grad = model.tcow.seeker.tracker_backbone.timesformer.model.patch_embed.proj.weight.grad
                head_grad = model.tcow.seeker.tracker_post_linear.weight.grad
                flow_grad = model.flow.output.weight.grad
                if args.smoke_steps:
                    expected = [patch_grad, head_grad]
                    if action_phase:
                        expected.append(flow_grad)
                        if args.wrist_camera and not args.freeze_wrist_encoder:
                            expected.append(model.flow.wrist_encoder.patch.weight.grad)
                    if any(g is None or not torch.isfinite(g).all() or not g.abs().any()
                           for g in expected):
                        raise ValueError("missing, zero, or nonfinite smoke gradient")
                if args.freeze_wrist_encoder and any(p.grad is not None for p in model.flow.wrist_encoder.parameters()):
                    raise ValueError("frozen wrist encoder created gradients")
                decoder_grad = None
                if args.contextualize and action_phase:
                    decoder_grad = sum(float(p.grad.float().abs().sum())
                                       for p in model.flow.context_decoder.parameters() if p.grad is not None)
                    if not np.isfinite(decoder_grad) or decoder_grad <= 0:
                        raise ValueError("missing or nonfinite context decoder gradient")
                print(json.dumps({"event": "gradient_smoke", "latent_shape": list(latent.shape),
                                  "action_phase": action_phase,
                                  "context_shape": list(context.shape) if has_action.any() else None,
                                  "patch_grad": float(patch_grad.float().abs().sum()) if patch_grad is not None else None,
                                  "mask_head_grad": float(head_grad.float().abs().sum()) if head_grad is not None else None,
                                  "flow_grad": float(flow_grad.float().abs().sum()) if flow_grad is not None else None,
                                  "context_decoder_grad": decoder_grad,
                                  "loss_action": float(action_loss), "loss_mask": float(mask_loss),
                                  "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else None}), flush=True)
                checked_phases.add(action_phase)
                faulthandler.cancel_dump_traceback_later()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config["train_args"].gradient_clip if config else 1.0,
                                           error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if args.smoke_steps:
                torch.cuda.synchronize() if device == "cuda" else torch.mps.synchronize()
                duration = time.perf_counter() - step_started
                print(json.dumps({"event": "smoke_step", "step": step,
                                  "action_phase": action_phase, "batch_size": rgbd.shape[0],
                                  "step_s": duration, "samples_per_s": rgbd.shape[0] / duration,
                                  "loss_action": float(action_loss), "loss_mask": float(mask_loss),
                                  "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30
                                  if device == "cuda" else None}), flush=True)
            progress.update(1)
            last_action_loss, last_mask_loss = float(action_loss), float(mask_loss)
            progress.set_postfix(action=f"{last_action_loss:.3f}",
                                 mask=f"{last_mask_loss:.3f}",
                                 chunks=int(has_action.sum()), refresh=False)
            del rgbd, query, truth, proprio, action, valid_steps, wrist, logits, latent
            del action_loss, mask_loss, total
            if action_phase:
                del noisy, tau, target_velocity, velocity, context, squared, weights
            if step in epoch_ends:
                save_checkpoint(args.output / "latest.pt", model, step, args)
                record = {"step": step, "clusters_done": (epoch_ends.index(step) + 1) * len(clusters),
                          "epoch": epoch_ends.index(step) + 1,
                          "action_loss": last_action_loss, "mask_loss": last_mask_loss,
                          "elapsed_s": time.perf_counter() - started,
                          "prefetch_wait_s": wait_s,
                          "compressed_video_gib": compressed_gib,
                          "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else None,
                          "peak_cpu_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
                history.append(record)
                (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                tqdm.write(json.dumps(record), file=sys.stdout)
    del iterator, dataloader
    release_videos(videos)
    (args.output / "final.pt").hardlink_to(args.output / "latest.pt")
    print(json.dumps({"event": "complete", "steps": step,
                      "elapsed_s": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
