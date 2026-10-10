"""Train OAT-Flow with H25 action chunks and optional TCOW mask supervision."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from OATFlow.policy.data import (
    policy_statistics, require_wrist_data, sample_index, split_rows,
)
from OATFlow.policy.model import (
    FlowMatchingHead, OATFlowPolicy, checkpoint_uses_depth, load_training_tracker, set_tracker_input,
)
from OATFlow.policy.loader import (
    ActionChunkDataset, cache_videos, make_dataloader, release_videos, training_batches, split_training_batches,
)
from OATFlow.policy.tracking import (
    MyLosses, Seeker, checkpoint_transformer_blocks, original_mask_loss,
)


def save_checkpoint(path, model, step, args, epoch):
    payload = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "step": step, "epoch": epoch, "visual_input": "rgb", "tcow_input_channels": 4,
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
               "context_initialization": "prior" if args.prior_checkpoint else "random",
               "source_prior_checkpoint": str(args.prior_checkpoint) if args.prior_checkpoint else None,
               "tcow_frozen": not any(p.requires_grad for p in model.tcow.parameters()),
               "flow_type": model.flow.flow_type,
               "action_representation": model.flow.action_representation,
               "action_dim": model.flow.action_dim,
               "action_input_dim": model.flow.action_dim, "action_output_dim": model.flow.action_dim,
               "action_padding": False,
               "source_flow_checkpoint": str(args.flow_weights) if args.flow_weights else None,
               "source_checkpoint": str(args.weights), "metrics": {},
               "validation_enabled": False, "checkpoint_selection": "latest_epoch",
               "mask_loss_weight": args.mask_loss_weight}
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    initialization = p.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--prior-checkpoint", type=Path, help="trained absolute-joint vision/FM prior")
    initialization.add_argument("--flow-weights", type=Path, help="SmolVLA action expert.safetensors for fresh FM")
    p.add_argument("--config-checkpoint", type=Path,
                   help="required for joint TCOW + flow training")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cluster-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int,
                   help="default: 8 on CUDA, 1 on MPS")
    p.add_argument("--micro-batch-size", type=int, help="physical batch; gradients accumulate to --batch-size")
    p.add_argument("--feature-cache", type=Path,
                   help="pre-adapter TCOW/wrist cache; requires both encoders frozen")
    p.add_argument("--tcow-lr", type=float, default=2e-5)
    p.add_argument("--flow-lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=3e-6,
                   help="FM LR floor; TCOW follows the same relative cosine decay")
    p.add_argument("--warmup-steps", type=int, default=215)
    p.add_argument("--mask-loss-weight", type=float, default=.2)
    p.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    p.add_argument("--flow-only", action="store_true",
                   help="freeze TCOW and optimize only the action flow")
    p.add_argument("--wrist-weights", type=Path,
                   help="torchvision ViT-B/16 ImageNet-1K V1 state dict for wrist initialization")
    p.add_argument("--freeze-wrist-encoder", action="store_true",
                   help="freeze the wrist ViT; wrist projection and FM remain trainable")
    p.set_defaults(wrist_camera=True, contextualize=True)
    p.add_argument("--seed", type=int, default=0)
    return p


def initialize_flow(args, tcow, statistics):
    flow = FlowMatchingHead(wrist_backbone=tcow.seeker.tracker_backbone, contextualize=True)
    if args.prior_checkpoint:
        if args.wrist_weights:
            raise ValueError("prior already supplies wrist weights")
        imported = flow.load_prior(args.prior_checkpoint)
        statistics = imported["normalization"]
    else:
        imported = flow.load_pretrained(args.flow_weights)
        flow.set_statistics(statistics)
        imported["context_initialization"] = {"initialization": "random", "layers": 4, "width": 960}
        if args.wrist_weights:
            imported["wrist_initialization"] = flow.wrist_encoder.load_vit_base(args.wrist_weights)
    if args.freeze_wrist_encoder:
        flow.wrist_encoder.freeze()
    imported["wrist_encoder_frozen"] = args.freeze_wrist_encoder
    return flow, imported, statistics


def main():
    args = argument_parser().parse_args()
    if args.flow_only:
        args.mask_loss_weight = 0.0
    if args.feature_cache and (not args.flow_only or not args.freeze_wrist_encoder or args.prior_checkpoint):
        raise ValueError("feature cache requires frozen encoders and explicit fresh FM/wrist sources")
    if args.wrist_weights and not args.wrist_weights.is_file():
        raise FileNotFoundError(args.wrist_weights)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    if not args.flow_only and args.config_checkpoint is None:
        raise ValueError("--config-checkpoint is required for joint training")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    for path in (args.flow_weights, args.prior_checkpoint):
        if path and not path.is_file():
            raise FileNotFoundError(path)
    if args.config_checkpoint and not args.config_checkpoint.is_file():
        raise FileNotFoundError(args.config_checkpoint)
    device = args.device
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
    if args.batch_size is None:
        args.batch_size = 1 if device == "mps" else 8
    if args.micro_batch_size is None:
        args.micro_batch_size = args.batch_size
    if min(args.cluster_size, args.epochs, args.batch_size) < 1:
        raise ValueError("cluster size, epochs, and batch size must be positive")
    if not 1 <= args.micro_batch_size <= args.batch_size:
        raise ValueError("micro batch must be 1..batch size")
    if not 0 < args.min_lr <= args.flow_lr or args.tcow_lr <= 0 or args.warmup_steps < 1:
        raise ValueError("positive LR floor <= peak and positive warmup required")
    train_rows, _ = split_rows(args.data)
    feature_cache = None
    if args.feature_cache:
        from OATFlow.policy.features import FeatureDataset, load_feature_cache
        feature_cache = load_feature_cache(args.feature_cache, args, train_rows)
    train_parents = len(train_rows)
    if args.wrist_camera:
        require_wrist_data(args.data, train_rows)
    rng = np.random.default_rng(args.seed)
    train_rows = [train_rows[i] for i in rng.permutation(len(train_rows))]
    clusters = [train_rows[i:i + args.cluster_size]
                for i in range(0, len(train_rows), args.cluster_size)]
    if args.output.exists():
        raise FileExistsError(args.output)
    statistics = policy_statistics(args.data, [row for rows in clusters for row in rows])
    args.output.mkdir(parents=True)
    (args.output / "config.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sample_fps": 25, "context_frames": 30, "action_horizon": 25,
        "stride": 1, "inference_execution_steps": 10,
        "flow_steps": 10, "flow_type": "dense_action_expert",
        "flow_time": "beta_sinusoidal_noise_at_t0",
        "train_episodes": sum(map(len, clusters)),
        "train_parent_episodes": sum(map(len, clusters)),
        "available_train_parent_episodes": train_parents,
        "validation_episodes": 0,
        "validation_enabled": False,
        "visual_input": "rgb", "tcow_input_channels": 4,
        "contextualize": args.contextualize,
        "context_tokens": 365,
        "action_representation": statistics.get("action_representation", "absolute_joint"),
        "action_dim": len(statistics["action_mean"]),
        "action_input_dim": 6, "action_output_dim": 6, "action_padding": False,
        "effective_batch_size": args.batch_size, "physical_batch_size": args.micro_batch_size,
        "lr_scheduler": "cosine_decay_with_warmup", "warmup_steps": args.warmup_steps,
        "video_loading": "frozen_feature_cache" if args.feature_cache else "lerobot_torchcodec_ram",
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
    source_step = load_training_tracker(tcow, pretrained)
    if not args.flow_only:
        checkpoint_transformer_blocks(tcow)
    flow, imported, statistics = initialize_flow(args, tcow, statistics)
    imported["action_representation"] = flow.action_representation
    imported["tcow_input"] = {"visual_input": "rgb", "patch_channels": 4,
                              "source_patch_channels": 5 if checkpoint_uses_depth(pretrained) else 4,
                              "conversion": "keep RGB and query kernels; discard depth kernel"}
    (args.output / "flow_initialization.json").write_text(
        json.dumps({"import": imported, "normalization": statistics}, indent=2) + "\n")
    print(json.dumps({"event": "pretrained_flow", **imported}), flush=True)
    (args.output / "README.md").write_text(
        f"# {args.output.name}\n\nTCOW + absolute-joint FM training.\n"
        f"Data {args.data}; TCOW weights {args.weights}; config {args.config_checkpoint}.\n"
        f"FM prior {args.prior_checkpoint}; fresh action expert {args.flow_weights}.\n"
        f"Epochs {args.epochs}; effective batch {args.batch_size}; micro batch {args.micro_batch_size}. "
        f"Warmup {args.warmup_steps} updates, cosine FM LR {args.flow_lr}→{args.min_lr}; "
        f"TCOW LR {args.tcow_lr} follows the same relative decay.\n"
        f"TCOW frozen {args.flow_only}; wrist frozen {args.freeze_wrist_encoder}; mask weight {args.mask_loss_weight}.\n"
        f"Feature cache {args.feature_cache}; encoders remain on CPU and skip forward when caching is enabled.\n"
        "30-frame TCOW history; 365 decoder tokens; H25 absolute actions, "
        "stride1, 25Hz; Euler10/K10 for evaluation.\n"
        "Prior normalization retained when transferring. Test: not run. Config: config.json, flow_initialization.json. Status: training.\n")
    model = OATFlowPolicy(tcow, flow)
    if args.feature_cache:
        # Encoders remain available in the saved policy, but never run in cached training.
        wrist_encoder = model.flow.wrist_encoder
        del model.flow.wrist_encoder
        model.flow.to(device)
        model.flow.wrist_encoder = wrist_encoder
    else:
        model.to(device)
    if args.flow_only:
        model.tcow.requires_grad_(False)
    losses = (MyLosses(config["train_args"], logging.getLogger("tcow"), "train")
              if config else None)
    groups = [{"params": [p for p in model.flow.parameters() if p.requires_grad], "lr": args.flow_lr,
               "betas": (.9, .95), "weight_decay": 1e-10}]
    if not args.flow_only:
        groups.insert(0, {"params": model.tcow.parameters(), "lr": args.tcow_lr,
                          "betas": (.9, .95), "weight_decay": 1e-10})
    optimizer = torch.optim.AdamW(groups)
    # The exact step count is known after indexing action_valid, before video decoding.
    # Keep the original cluster/sample order; only image loading changes.
    plans = [[[(end, has_action) for end, has_action in sample_index(args.data / row["path"])
               if has_action or not args.flow_only] for row in rows] for rows in
             tqdm(clusters, desc="index episodes", unit="cluster", file=sys.stdout)]
    samples, batches, epoch_ends = training_batches(
        plans, args.epochs, args.batch_size, rng)
    total_steps = len(batches)
    total_samples = len(samples)
    total_action = sum(active for _episode, _end, active in samples)
    if args.warmup_steps >= total_steps:
        raise ValueError("warmup must end before final training step")
    scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=args.warmup_steps, num_decay_steps=total_steps,
        peak_lr=args.flow_lr, decay_lr=args.min_lr).build(optimizer, num_training_steps=total_steps)
    flat_rows = [row for rows in clusters for row in rows]
    physical_batches, batch_weights = split_training_batches(batches, samples, flat_rows, args.micro_batch_size)
    run_config = json.loads((args.output / "config.json").read_text())
    run_config.update(optimizer_steps=total_steps, physical_batches=len(physical_batches),
                      samples_per_epoch=total_samples, action_chunks_per_epoch=total_action,
                      context_samples_per_epoch=total_samples-total_action,
                      optimizer_epoch_ends=epoch_ends)
    (args.output / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    videos, compressed_gib = None, 0.
    if args.feature_cache:
        dataset = FeatureDataset(feature_cache, flat_rows, samples)
    else:
        videos, compressed_gib = cache_videos(args.data, flat_rows, args.wrist_camera)
        print(json.dumps({"event": "video_cache", "videos": len(videos),
                          "compressed_gib": compressed_gib, "video_decoder_cache_size": 8,
                          "backend": "lerobot/torchcodec",
                          "num_workers": 2, "prefetch_factor": 2,
                          "prefetched_batches": 4}), flush=True)
        cache_directory = str(Path(next(iter(videos.values()))).parent)
        (args.output / "video_cache.json").write_text(json.dumps({
            "directory": cache_directory, "compressed_gib": compressed_gib,
            "shared_with_workers": True, "filesystem": "tmpfs"}, indent=2) + "\n")
        dataset = ActionChunkDataset(args.data, flat_rows, samples, videos, args.flow_only, args.wrist_camera)
    dataloader = make_dataloader(dataset, physical_batches, args.seed, device == "cuda")
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
                      "micro_batch_size": args.micro_batch_size, "physical_batches": len(physical_batches),
                      "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "Apple MPS",
                      "flow_only": args.flow_only,
                      "source_tcow_step": source_step,
                      "fresh_flow": args.prior_checkpoint is None,
                      "flow_type": model.flow.flow_type}), flush=True)
    step = 0
    history = []
    wait_s = 0.0
    checked_phases = set()
    last_micro_log = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    iterator = iter(dataloader)
    try:
        with tqdm(total=total_steps, desc="train TCOW + flow", unit="step", mininterval=5,
                  file=sys.stdout) as progress:
            for batch_index, (logical_size, logical_valid, first_micro, last_micro, consumed) in enumerate(batch_weights):
                if first_micro:
                    optimizer.zero_grad(set_to_none=True)
                    logical_action_loss, logical_mask_loss = 0., 0.
                    observed_valid = 0
                waiting = time.perf_counter()
                tensors = next(iterator)
                wait_s += time.perf_counter() - waiting
                tensors = [tensor.to(device, non_blocking=device == "cuda") for tensor in tensors]
                wrist_features = None
                if args.feature_cache:
                    latent, proprio, action, valid_steps, wrist_features = tensors
                    rgbd = query = truth = wrist = None
                else:
                    rgbd, query, truth, proprio, action, valid_steps = tensors[:6]
                    wrist = tensors[6] if args.wrist_camera else None
                del tensors
                has_action = valid_steps.any(dim=1)
                model.train()
                if args.flow_only:
                    model.tcow.eval()
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    if args.feature_cache:
                        logits = latent.new_zeros(())
                    else:
                        with torch.set_grad_enabled(not args.flow_only):
                            logits, _ = model.tcow(rgbd, query)
                        latent = model._latent
                    if args.flow_only and not args.feature_cache:
                        # Frozen tracking only supplies features; its full mask
                        # output need not stay allocated during FM backward.
                        del logits
                        logits = latent.new_zeros(())
                    if has_action.any():
                        noisy, tau, target_velocity = model.flow.training_path(action[has_action])
                        velocity, context = model.flow(
                            latent[has_action], proprio[has_action], noisy, tau,
                            wrist=wrist[has_action] if wrist is not None else None,
                            wrist_features=wrist_features[has_action] if wrist_features is not None else None)
                        squared = (velocity.float() - target_velocity).square()
                        weights = valid_steps[has_action, :, None]
                        action_loss = (squared * weights).sum() / (weights.sum() * action.shape[-1])
                    else:
                        action_loss = logits.new_zeros(())
                mask_loss = (logits.new_zeros(()) if args.flow_only else
                             original_mask_loss(losses, logits, truth, step / max(total_steps, 1)))
                micro_valid = int(valid_steps.sum())
                observed_valid += micro_valid
                action_scale = micro_valid / max(logical_valid, 1)
                mask_scale = len(proprio) / logical_size
                total = action_loss * action_scale + args.mask_loss_weight * mask_loss * mask_scale
                if not torch.isfinite(total):
                    raise ValueError("nonfinite training loss")
                total.backward()
                logical_action_loss += float(action_loss) * action_scale
                logical_mask_loss += float(mask_loss) * mask_scale
                model._latent = None
                action_phase = bool(has_action.any())
                if action_phase not in checked_phases:
                    patch_grad = model.tcow.seeker.tracker_backbone.timesformer.model.patch_embed.proj.weight.grad
                    head_grad = model.tcow.seeker.tracker_post_linear.weight.grad
                    flow_grad = model.flow.output.weight.grad
                    expected = [] if args.flow_only else [patch_grad]
                    if args.mask_loss_weight > 0 and not args.flow_only:
                        expected.append(head_grad)
                    if action_phase:
                        expected.append(flow_grad)
                    if any(g is None or not torch.isfinite(g).all() or not g.abs().any() for g in expected):
                        raise ValueError("missing, zero or nonfinite trainable gradient")
                    if args.freeze_wrist_encoder and any(p.grad is not None for p in model.flow.wrist_encoder.parameters()):
                        raise ValueError("frozen wrist encoder created gradients")
                    decoder_grad = None
                    if args.contextualize and action_phase:
                        decoder_grad = sum(float(p.grad.float().abs().sum())
                                           for p in model.flow.context_decoder.parameters() if p.grad is not None)
                        if not np.isfinite(decoder_grad) or decoder_grad <= 0:
                            raise ValueError("missing or nonfinite context decoder gradient")
                    print(json.dumps({"event": "gradients_checked", "latent_shape": list(latent.shape),
                                      "action_phase": action_phase,
                                      "context_shape": list(context.shape) if has_action.any() else None,
                                      "patch_grad": float(patch_grad.float().abs().sum()) if patch_grad is not None else None,
                                      "mask_head_grad": float(head_grad.float().abs().sum()) if head_grad is not None else None,
                                      "flow_grad": float(flow_grad.float().abs().sum()) if flow_grad is not None else None,
                                      "context_decoder_grad": decoder_grad,
                                      "loss_action": float(action_loss), "loss_mask": float(mask_loss),
                                      "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else None}), flush=True)
                    checked_phases.add(action_phase)
                if last_micro:
                    if observed_valid != logical_valid:
                        raise ValueError("microbatch action validity differs from logical batch normalization")
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   config["train_args"].gradient_clip if config else 1.0,
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    progress.update(1)
                last_action_loss, last_mask_loss = logical_action_loss, logical_mask_loss
                progress.set_postfix(action=f"{last_action_loss:.3f}", mask=f"{last_mask_loss:.3f}",
                                     accumulated=f"{consumed}/{logical_size}",
                                     lr=f"{optimizer.param_groups[-1]['lr']:.2e}", refresh=False)
                if time.perf_counter() - last_micro_log > 10:
                    progress.refresh()
                    last_micro_log = time.perf_counter()
                del rgbd, query, truth, proprio, action, valid_steps, wrist, wrist_features, logits, latent
                del action_loss, mask_loss, total
                if action_phase:
                    del noisy, tau, target_velocity, velocity, context, squared, weights
                if last_micro and step in epoch_ends:
                    epoch = epoch_ends.index(step) + 1
                    save_checkpoint(args.output / "latest.pt", model, step, args, epoch)
                    record = {"step": step, "clusters_done": (epoch_ends.index(step) + 1) * len(clusters),
                              "epoch": epoch_ends.index(step) + 1,
                              "action_loss": last_action_loss, "mask_loss": last_mask_loss,
                              "flow_lr": optimizer.param_groups[-1]["lr"],
                              "tcow_lr": optimizer.param_groups[0]["lr"] if not args.flow_only else None,
                              "elapsed_s": time.perf_counter() - started,
                              "prefetch_wait_s": wait_s,
                              "compressed_video_gib": compressed_gib,
                              "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30 if device == "cuda" else None,
                              "peak_cpu_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
                    history.append(record)
                    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                    tqdm.write(json.dumps(record), file=sys.stdout)
    finally:
        del iterator, dataloader
        if videos is not None:
            release_videos(videos)
    (args.output / "final.pt").hardlink_to(args.output / "latest.pt")
    readme = args.output / "README.md"
    readme.write_text(readme.read_text().replace("Status: training", "Status: complete"))
    print(json.dumps({"event": "complete", "steps": step,
                      "elapsed_s": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
