"""Train the same joint policy with torchrun on up to three GPUs."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import timedelta
import faulthandler
import json
import logging
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
from threading import Lock

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from OATFlow.policy.data import (
    expand_demonstrations, load_joint_episode, load_policy_episode, policy_statistics, require_wrist_data, sample_index, split_rows, training_clip,
    training_samples,
)
from OATFlow.policy.model import (
    FlowMatchingHead, OATFlowPolicy, checkpoint_uses_depth, load_training_tracker, set_tracker_input,
)
from OATFlow.policy.tracking import (
    MyLosses, Seeker, checkpoint_transformer_blocks, original_mask_loss,
)
from OATFlow.policy.train import (
    argument_parser, make_optimizer, make_scheduler, save_checkpoint,
)


def rank_batch(batch, sizes, rank):
    start = sum(sizes[:rank])
    items = batch[start:start + sizes[rank]]
    # A short final batch still needs one forward/backward on every rank.
    # Its dummy sample gets zero weight, so no training sample is duplicated.
    return (items if items else batch[:1]), len(items)


def cache_cluster(data_root, rows, destination, flow_only=False, require_wrist=False):
    """Decode once into temporary RAM files that every rank can map."""
    destination.mkdir()
    occupied = sum(p.stat().st_size for p in destination.parent.rglob("*.npy"))
    lock = Lock()

    def cache_one(item):
        nonlocal occupied
        index, row = item
        episode = (load_policy_episode if flow_only else load_joint_episode)(
            data_root / row["path"], require_wrist=require_wrist, training_cache=True)
        arrays, scalars = {}, {}
        for key, value in episode.items():
            if isinstance(value, np.ndarray):
                with lock:
                    occupied += value.nbytes
                    if occupied > 28 * 2**30:
                        raise MemoryError("shared episode buffers exceed 28 GiB; reduce cluster size")
                path = destination / f"{index}_{key}.npy"
                mapped = np.lib.format.open_memmap(path, mode="w+", dtype=value.dtype, shape=value.shape)
                mapped[:] = value
                mapped.flush()
                mapped._mmap.close()
                arrays[key] = str(path)
            else:
                scalars[key] = value
        return row, scalars, arrays

    with ThreadPoolExecutor(max_workers=min(4, len(rows))) as decoders:
        descriptors = list(tqdm(decoders.map(cache_one, enumerate(rows)), total=len(rows),
                                desc="load shared episodes", unit="episode", mininterval=5,
                                file=sys.stdout))
    tqdm.write(json.dumps({"event": "shared_cluster_ready", "episodes": len(rows),
                           "shared_buffers_gib": occupied / 2**30}), file=sys.stdout)
    return descriptors


def release_cluster(episodes, destination, rank):
    for _row, episode in episodes:
        for value in episode.values():
            if isinstance(value, np.memmap):
                value._mmap.close()
    dist.barrier()
    if rank == 0:
        shutil.rmtree(destination)
    dist.barrier()


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argument_parser(distributed=True)
    parser.description = __doc__
    parser.add_argument("--rank-batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--resume", type=Path,
                        help="continue joint model weights from an epoch checkpoint; AdamW restarts")
    args = parser.parse_args()
    if (args.wrist_weights or args.freeze_wrist_encoder) and not args.wrist_camera:
        raise ValueError("wrist initialization/freezing requires --wrist-camera")
    if args.wrist_weights and not args.wrist_weights.is_file():
        raise FileNotFoundError(args.wrist_weights)
    if args.contextualize:
        if not (args.wrist_camera and args.wrist_weights):
            raise ValueError("contextualization requires a pretrained wrist ViT")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.device != "cuda" or not 2 <= world <= 3:
        raise ValueError("torchrun must use two or three CUDA GPUs")
    if len(args.rank_batch_sizes) != world or min(args.rank_batch_sizes) < 1:
        raise ValueError("supply one positive batch size per rank")
    if min(args.epochs, args.cluster_size) < 1:
        raise ValueError("epochs and cluster size must be positive")
    if args.config_checkpoint is None:
        raise ValueError("--config-checkpoint is required")
    for path in (args.weights, args.flow_weights, args.config_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.resume and not args.resume.is_file():
        raise FileNotFoundError(args.resume)
    args.batch_size = sum(args.rank_batch_sizes)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=2),
                            device_id=torch.device("cuda", local_rank))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    train_rows, _ = split_rows(args.data)
    train_parents = len(train_rows)
    train_rows = ([{**row, "parent_path": row["path"], "demonstration_id": "base"} for row in train_rows]
                  if args.base_demonstrations_only else expand_demonstrations(args.data, train_rows))
    if args.wrist_camera:
        require_wrist_data(args.data, train_rows)
    train_rows = [train_rows[i] for i in rng.permutation(len(train_rows))]
    clusters = [train_rows[i:i + args.cluster_size]
                for i in range(0, len(train_rows), args.cluster_size)]
    if args.max_clusters:
        clusters = clusters[:args.max_clusters]
    if args.smoke_steps > len(clusters) * args.epochs * 4:
        raise ValueError("eight-step smoke requires at least two clusters")
    if rank == 0:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        (args.output / "config.json").write_text(json.dumps({
            **{k: str(v) if hasattr(v, "__fspath__") else v for k, v in vars(args).items()},
            "world_size": world, "sample_fps": 25, "context_frames": 30,
            "action_horizon": 25, "context_stride": 10, "action_stride": 10,
            "inference_execution_steps": 10, "flow_steps": 10,
            "flow_type": "dense_action_expert", "flow_time": "beta_sinusoidal_noise_at_t0",
            "train_episodes": sum(map(len, clusters)),
            "train_parent_episodes": len({row["parent_path"] for rows in clusters for row in rows}),
            "available_train_parent_episodes": train_parents,
            "validation_episodes": 0,
            "validation_enabled": False,
            "visual_input": "rgb", "tcow_input_channels": 4,
            "contextualize": args.contextualize,
            "context_tokens": 365 if args.contextualize else (701 if args.wrist_camera else 301),
        }, indent=2) + "\n")
    dist.barrier()
    config = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    source = torch.load(args.weights, map_location="cpu", weights_only=False, mmap=True)
    seeker_args = dict(config["seeker_args"])
    seeker_args["tracker_pretrained"] = False
    tcow = set_tracker_input(Seeker(logging.getLogger("tcow"), **seeker_args))
    source_step = load_training_tracker(tcow, source, args.flow_only)
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
    statistics = [policy_statistics(args.data, [r for rows in clusters for r in rows])
                  if rank == 0 else None]
    dist.broadcast_object_list(statistics, src=0)
    flow.relative_actions = statistics[0].get("action_representation") == "relative_joint"
    flow.set_statistics(statistics[0])
    if "model" in source:
        flow.visual.load_state_dict({k.removeprefix("flow.visual."): v for k, v in source["model"].items()
                                     if k.startswith("flow.visual.")}, strict=True)
        imported["visual_initialization"] = str(args.weights)
    imported["action_representation"] = flow.action_representation
    imported["tcow_input"] = {"visual_input": "rgb", "patch_channels": 4,
                              "source_patch_channels": 5 if checkpoint_uses_depth(source) else 4,
                              "conversion": "keep RGB and query kernels; discard depth kernel"}
    model = OATFlowPolicy(tcow, flow)
    resume_step = 0
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        if checkpoint_uses_depth(resumed):
            raise ValueError("RGB training cannot resume an RGB-D run; initialize a fresh experiment")
        if (resumed["flow_type"] != model.flow.flow_type or
                resumed["wrist_camera"] != args.wrist_camera or
                bool(resumed.get("contextualize", False)) != flow.contextualize or
                resumed["action_representation"] != imported["action_representation"]):
            raise ValueError("resume checkpoint architecture or action representation differs")
        if args.wrist_camera and bool(resumed.get("wrist_imagenet_normalization", False)) != flow.wrist_encoder.imagenet_normalization:
            raise ValueError("resume checkpoint wrist normalization differs")
        model.load_state_dict(resumed["model"], strict=True)
        resume_step = int(resumed["step"])
        imported.update(resume_checkpoint=str(args.resume), resume_step=resume_step,
                        resumed_tensors=len(resumed["model"]), optimizer_restored=False)
        del resumed
    if rank == 0:
        (args.output / "flow_initialization.json").write_text(
            json.dumps({"import": imported, "normalization": statistics[0]}, indent=2) + "\n")
        print(json.dumps({"event": "pretrained_flow", **imported}), flush=True)
    model = model.cuda()
    if args.flow_only:
        model.tcow.requires_grad_(False)
    # FM is unused on mask-only batches; the used parameter set changes each step.
    engine = DistributedDataParallel(model, device_ids=[local_rank],
                                     find_unused_parameters=not args.flow_only, broadcast_buffers=False,
                                     gradient_as_bucket_view=True)
    torch.manual_seed(args.seed + rank)
    losses = MyLosses(config["train_args"], logging.getLogger("tcow"), "train")
    optimizer = make_optimizer(model, args)
    plans = [[[(end, action) for end, action in sample_index(args.data / row["path"])
               if action or not args.flow_only] for row in rows] for rows in
             tqdm(clusters, desc="index episodes", unit="cluster", file=sys.stdout, disable=rank != 0)]
    steps_per_epoch = sum(
        sum((count + args.batch_size - 1) // args.batch_size for count in
            (sum(not action for plan in cluster for _, action in plan),
             sum(action for plan in cluster for _, action in plan))) for cluster in plans)
    total_steps = args.epochs * steps_per_epoch
    scheduler = make_scheduler(optimizer, args, total_steps)
    if resume_step % steps_per_epoch:
        raise ValueError("resume requires an epoch boundary with the same batching")
    completed_epochs = resume_step // steps_per_epoch
    total_samples = sum(len(plan) for cluster in plans for plan in cluster)
    total_action = sum(action for cluster in plans for plan in cluster for _, action in plan)
    if rank == 0:
        print(json.dumps({"event": "start", "steps": total_steps, "epochs": args.epochs,
                          "samples_per_epoch": total_samples, "mask_only_per_epoch": total_samples - total_action,
                          "action_chunks_per_epoch": total_action, "batch_size": args.batch_size,
                          "rank_batch_sizes": args.rank_batch_sizes, "world_size": world,
                          "device_name": torch.cuda.get_device_name(), "source_tcow_step": source_step,
                          "trainable_tcow": sum(p.numel() for p in model.tcow.parameters() if p.requires_grad),
                          "trainable_flow": sum(p.numel() for p in model.flow.parameters() if p.requires_grad),
                          "trainable_wrist_encoder": sum(p.numel() for p in model.flow.wrist_encoder.parameters()
                                                         if p.requires_grad) if args.wrist_camera else 0,
                          "lr_schedule": args.lr_schedule,
                          "warmup_steps": max(1, int(total_steps * args.warmup_fraction)) if args.warmup_fraction else 0,
                          "peak_lrs": {g["name"]: g["initial_lr"] for g in optimizer.param_groups},
                          "resume_step": resume_step, "completed_epochs": completed_epochs,
                          "total_epochs": completed_epochs + args.epochs,
                          "fresh_flow": False, "flow_type": model.flow.flow_type}), flush=True)
    step, wait_s = 0, 0.0
    history, checked_phases = [], set()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    shared_root = Path("/dev/shm") / args.output.name
    if rank == 0:
        shared_root.mkdir(mode=0o700)
    # Video decoding must not share Python's interpreter lock with rank 0's training loop.
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as loader:
        future = (loader.submit(cache_cluster, args.data, clusters[0], shared_root / "0", args.flow_only,
                                args.wrist_camera)
                  if rank == 0 else None)
        with tqdm(total=args.smoke_steps or total_steps, desc="train TCOW + flow (DDP)",
                  unit="step", mininterval=5, file=sys.stdout, disable=rank != 0) as progress:
            for cluster_index in range(len(clusters) * args.epochs):
                cluster_id = cluster_index % len(clusters)
                waiting = time.perf_counter()
                publication = [future.result() if rank == 0 else None]
                dist.broadcast_object_list(publication, src=0)
                episodes = [(row, {**scalars, **{key: np.load(path, mmap_mode="r")
                                               for key, path in arrays.items()}})
                            for row, scalars, arrays in publication[0]]
                wait_s += time.perf_counter() - waiting
                future = (loader.submit(cache_cluster, args.data,
                                        clusters[(cluster_index + 1) % len(clusters)],
                                        shared_root / str(cluster_index + 1), args.flow_only, args.wrist_camera)
                          if rank == 0 and
                          (not args.smoke_steps or (cluster_index + 1) * 4 < args.smoke_steps) and
                          cluster_index + 1 < len(clusters) * args.epochs else None)
                items = [(i, end, action) for i, plan in enumerate(plans[cluster_id]) for end, action in plan]
                for i, (_row, episode) in enumerate(episodes):
                    actual = [(end, action) for end, action in training_samples(episode)
                              if action or not args.flow_only]
                    if actual != plans[cluster_id][i]:
                        raise ValueError(f"indexed samples changed: {episode['name']}")
                phases = []
                for action_phase in (False, True):
                    phase_items = [item for item in items if item[2] == action_phase]
                    rng.shuffle(phase_items)
                    phases.append([phase_items[i:i + args.batch_size]
                                   for i in range(0, len(phase_items), args.batch_size)])
                batches = phases[0] + phases[1]
                rng.shuffle(batches)
                if args.smoke_steps:
                    # Exercise both branches, including a partial tail batch with an empty rank.
                    batches = ([phases[1][0], phases[1][1], phases[1][-1], phases[1][-1]] if args.flow_only else
                               [phases[0][0], phases[1][0], phases[0][-1], phases[1][-1]])
                for batch in batches:
                    batch_items, count = rank_batch(batch, args.rank_batch_sizes, rank)
                    samples = [training_clip(episodes[i][1], end, action, "cuda", include_wrist=args.wrist_camera)
                               for i, end, action in batch_items]
                    tensors = [torch.cat(parts) for parts in zip(*samples)]
                    rgbd, query, truth, proprio, action, valid = tensors[:6]
                    wrist = tensors[6] if args.wrist_camera else None
                    del tensors
                    is_action = batch[0][2]
                    model.train()
                    if args.flow_only:
                        model.tcow.eval()
                    noisy, tau, target = model.flow.training_path(action) if is_action else (None, None, None)
                    # DDP averages gradients across ranks. Weight by the actual sample/valid-step
                    # counts, including short batches, to retain the global action-loss mean.
                    valid_count = valid.sum().float() * bool(count)
                    dist.all_reduce(valid_count)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, velocity, latent, context = engine(rgbd, query, proprio, noisy, tau, wrist=wrist)
                        action_loss = (((velocity.float() - target).square() * valid[:, :, None]).sum()
                                       * (world * bool(count)) / (valid_count * action.shape[-1])) if is_action else logits.new_zeros(())
                    mask_loss = (logits.sum() * 0 if args.flow_only else
                                 original_mask_loss(losses, logits, truth,
                                                    (resume_step + step) / max(resume_step + total_steps, 1))
                                 * (world * count / len(batch)))
                    total = action_loss + args.mask_loss_weight * mask_loss
                    if not torch.isfinite(total):
                        raise ValueError(f"nonfinite loss on rank {rank}")
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    model._latent = None
                    if is_action not in checked_phases:
                        patch = model.tcow.seeker.tracker_backbone.timesformer.model.patch_embed.proj.weight.grad
                        head = model.tcow.seeker.tracker_post_linear.weight.grad
                        out = model.flow.output.weight.grad
                        if args.flow_only and (patch is not None or head is not None or out is None or
                                               not torch.isfinite(out).all() or not out.abs().sum()):
                            raise AssertionError("invalid frozen TCOW / trainable FM gradients")
                        if not is_action and out is not None:
                            raise AssertionError("mask-only batch created FM gradients")
                        values = [float(p.float().abs().sum()) if p is not None else None
                                  for p in (patch, head, out)]
                        required = values if is_action else values[:2]
                        if not args.flow_only and any(v is None or not np.isfinite(v) or v <= 0 for v in required):
                            raise AssertionError(f"missing or nonfinite gradients on rank {rank}")
                        wrist_grad = model.flow.wrist_encoder.patch.weight.grad if args.wrist_camera else None
                        wrist_value = float(wrist_grad.float().abs().sum()) if wrist_grad is not None else None
                        if args.freeze_wrist_encoder and any(p.grad is not None for p in model.flow.wrist_encoder.parameters()):
                            raise AssertionError("frozen wrist encoder created gradients")
                        if args.wrist_camera and not args.freeze_wrist_encoder and is_action and (wrist_value is None or not np.isfinite(wrist_value) or wrist_value <= 0):
                            raise AssertionError(f"missing or nonfinite wrist gradient on rank {rank}")
                        if not is_action and wrist_grad is not None:
                            raise AssertionError("mask-only batch created wrist gradients")
                        if rank == 0:
                            print(json.dumps({"event": "gradient_smoke", "action_phase": is_action,
                                              "latent_shape": list(latent.shape),
                                              "context_shape": list(context.shape) if context is not None else None,
                                              "patch_grad": values[0], "mask_head_grad": values[1],
                                              "flow_grad": values[2], "wrist_grad": wrist_value,
                                              "frozen_wrist_encoder": args.freeze_wrist_encoder}), flush=True)
                        checked_phases.add(is_action)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["train_args"].gradient_clip,
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    logs = torch.stack((action_loss.detach(), mask_loss.detach()))
                    dist.all_reduce(logs)
                    logs /= world
                    progress.update()
                    progress.set_postfix(action=f"{logs[0].item():.3f}", mask=f"{logs[1].item():.3f}",
                                         chunks=len(batch) if is_action else 0, refresh=False)
                    if rank == 0 and step % 25 == 0:
                        elapsed = time.perf_counter() - started
                        print(json.dumps({"event": "training_progress", "step": step, "steps": total_steps,
                                          "elapsed_s": elapsed, "seconds_per_step": elapsed / step,
                                          "eta_hours": (total_steps - step) * elapsed / step / 3600,
                                          "lrs": {g["name"]: g["lr"] for g in optimizer.param_groups},
                                          "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30}), flush=True)
                    # Release RGB-D clips and full-video masks before allocating the next batch.
                    del samples, rgbd, query, truth, proprio, action, valid, wrist
                    del noisy, tau, target, logits, velocity, latent, context
                    del action_loss, mask_loss, total
                    if args.smoke_steps and step >= args.smoke_steps:
                        # Compare an updated action projection and TCOW patch checksum across ranks.
                        signature = torch.stack((model.flow.output.weight.detach().double().sum(),
                                                 model.tcow.seeker.tracker_backbone.timesformer.model.patch_embed.proj.weight.detach().double().sum()))
                        signatures = [torch.zeros_like(signature) for _ in range(world)]
                        dist.all_gather(signatures, signature)
                        for other in signatures[1:]:
                            torch.testing.assert_close(signatures[0], other, rtol=0, atol=0)
                        memory = torch.tensor([torch.cuda.max_memory_reserved() / 2**30,
                                               resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20], device="cuda")
                        memories = [torch.zeros_like(memory) for _ in range(world)]
                        dist.all_gather(memories, memory)
                        if rank == 0:
                            report = {"event": "smoke_complete", "steps": step, "world_size": world,
                                      "rank_batch_sizes": args.rank_batch_sizes, "synchronized": True,
                                      "expected_loss_phases_passed": checked_phases == ({True} if args.flow_only else {False, True}),
                                      "frozen_tcow": args.flow_only,
                                      "per_rank_peak_vram_cpu_rss_gib": [m.tolist() for m in memories]}
                            (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
                            print(json.dumps(report), flush=True)
                        release_cluster(episodes, shared_root / str(cluster_index), rank)
                        if rank == 0:
                            shared_root.rmdir()
                        dist.destroy_process_group()
                        return
                release_cluster(episodes, shared_root / str(cluster_index), rank)
                del episodes
                if (cluster_index + 1) % len(clusters) == 0:
                    dist.barrier()
                    if rank == 0:
                        save_checkpoint(args.output / "latest.pt", model, resume_step + step, args)
                        record = {"step": resume_step + step, "resumed_run_step": step,
                                  "clusters_done": completed_epochs * len(clusters) + cluster_index + 1,
                                  "epoch": completed_epochs + cluster_index // len(clusters) + 1,
                                  "action_loss": float(logs[0]), "mask_loss": float(logs[1]),
                                  "elapsed_s": time.perf_counter() - started, "prefetch_wait_s": wait_s,
                                  "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30,
                                  "peak_cpu_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
                        history.append(record)
                        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                        tqdm.write(json.dumps(record), file=sys.stdout)
                    dist.barrier()
    if rank == 0:
        shared_root.rmdir()
        (args.output / "final.pt").hardlink_to(args.output / "latest.pt")
        print(json.dumps({"event": "complete", "steps": resume_step + step, "resumed_run_steps": step,
                          "elapsed_s": time.perf_counter() - started}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
