"""Train the same joint policy with torchrun on up to three GPUs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import faulthandler
import json
import logging
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from memory_occlusion.experiments.tcow_joint_flow.data import (
    load_joint_episode, load_policy_episode, policy_statistics, sample_index, split_rows, training_clip,
    training_samples,
)
from memory_occlusion.experiments.tcow_joint_flow.flow_matching import DenseFlowMatching
from memory_occlusion.experiments.tcow_joint_flow.model import JointTCOWFlow, expand_depth_channel, load_training_tracker
from memory_occlusion.experiments.tcow_joint_flow.tcow import (
    MyLosses, Seeker, checkpoint_transformer_blocks, original_mask_loss,
)
from memory_occlusion.experiments.tcow_joint_flow.train_from_tcow_context import (
    argument_parser, evaluate, save_checkpoint,
)
from memory_occlusion.experiments.tcow_joint_flow.validation import evaluate_closed_loop


def rank_batch(batch, sizes, rank):
    start = sum(sizes[:rank])
    items = batch[start:start + sizes[rank]]
    # A short final batch still needs one forward/backward on every rank.
    # Its dummy sample gets zero weight, so no training sample is duplicated.
    return (items if items else batch[:1]), len(items)


def cache_cluster(data_root, rows, destination, flow_only=False):
    """Decode once into temporary RAM files that every rank can map."""
    destination.mkdir()
    occupied = sum(p.stat().st_size for p in destination.parent.rglob("*.npy"))
    descriptors = []
    for index, row in enumerate(tqdm(rows, desc="load shared episodes", unit="episode",
                                     mininterval=5, file=sys.stdout)):
        episode = (load_policy_episode if flow_only else load_joint_episode)(data_root / row["path"])
        arrays, scalars = {}, {}
        for key, value in episode.items():
            if isinstance(value, np.ndarray):
                occupied += value.nbytes
                if occupied > 48 * 2**30:
                    raise MemoryError("shared episode buffers exceed 48 GiB; reduce cluster size")
                path = destination / f"{index}_{key}.npy"
                mapped = np.lib.format.open_memmap(path, mode="w+", dtype=value.dtype, shape=value.shape)
                mapped[:] = value
                mapped.flush()
                mapped._mmap.close()
                arrays[key] = str(path)
            else:
                scalars[key] = value
        descriptors.append((row, scalars, arrays))
        del episode
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
    parser = argument_parser()
    parser.description = __doc__
    parser.add_argument("--rank-batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--smoke-steps", type=int, choices=(0, 4, 8), default=0,
                        help="verify gradients and synchronization; eight steps cross a cluster boundary")
    args = parser.parse_args()
    if args.closed_loop_every_evals < 0:
        raise ValueError("closed-loop-every-evals must be nonnegative")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.device != "cuda" or not 2 <= world <= 3:
        raise ValueError("torchrun must use two or three CUDA GPUs")
    if len(args.rank_batch_sizes) != world or min(args.rank_batch_sizes) < 1:
        raise ValueError("supply one positive batch size per rank")
    if min(args.epochs, args.cluster_size, args.eval_every_clusters) < 1:
        raise ValueError("epochs, cluster size and evaluation interval must be positive")
    if args.config_checkpoint is None:
        raise ValueError("--config-checkpoint is required")
    for path in (args.weights, args.flow_weights, args.config_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.batch_size = sum(args.rank_batch_sizes)
    torch.set_num_threads(2)
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=2),
                            device_id=torch.device("cuda", local_rank))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    train_rows, val_rows = split_rows(args.data)
    if args.max_val_episodes:
        val_rows = val_rows[:args.max_val_episodes]
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
            "train_episodes": sum(map(len, clusters)), "validation_episodes": len(val_rows),
        }, indent=2) + "\n")
    dist.barrier()
    config = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    source = torch.load(args.weights, map_location="cpu", weights_only=False, mmap=True)
    seeker_args = dict(config["seeker_args"])
    seeker_args["tracker_pretrained"] = False
    tcow = expand_depth_channel(Seeker(logging.getLogger("tcow"), **seeker_args))
    source_step = load_training_tracker(tcow, source, args.flow_only)
    if not args.flow_only:
        checkpoint_transformer_blocks(tcow)
    flow = DenseFlowMatching()
    imported = flow.load_pretrained(args.flow_weights)
    statistics = [policy_statistics(args.data, [r for rows in clusters for r in rows])
                  if rank == 0 else None]
    dist.broadcast_object_list(statistics, src=0)
    flow.relative_actions = statistics[0].get("action_representation") == "relative_joint"
    flow.set_statistics(statistics[0])
    if "model" in source:
        flow.visual.load_state_dict({k.removeprefix("flow.visual."): v for k, v in source["model"].items()
                                     if k.startswith("flow.visual.")}, strict=True)
        imported["visual_initialization"] = str(args.weights)
    imported["action_representation"] = "relative_joint" if flow.relative_actions else "absolute_joint"
    if rank == 0:
        (args.output / "flow_initialization.json").write_text(
            json.dumps({"import": imported, "normalization": statistics[0]}, indent=2) + "\n")
        print(json.dumps({"event": "pretrained_flow", **imported}), flush=True)
    model = JointTCOWFlow(tcow, flow).cuda()
    if args.flow_only:
        model.tcow.requires_grad_(False)
    # FM is unused on mask-only batches; the used parameter set changes each step.
    engine = DistributedDataParallel(model, device_ids=[local_rank],
                                     find_unused_parameters=not args.flow_only, broadcast_buffers=False,
                                     gradient_as_bucket_view=True)
    torch.manual_seed(args.seed + rank)
    losses = MyLosses(config["train_args"], logging.getLogger("tcow"), "train")
    groups = [{"params": model.flow.parameters(), "lr": args.flow_lr,
               "betas": (.9, .95), "weight_decay": 1e-10}]
    if not args.flow_only:
        groups.insert(0, {"params": model.tcow.parameters(), "lr": args.tcow_lr})
    optimizer = torch.optim.AdamW(groups)
    plans = [[[(end, action) for end, action in sample_index(args.data / row["path"])
               if action or not args.flow_only] for row in rows] for rows in
             tqdm(clusters, desc="index episodes", unit="cluster", file=sys.stdout, disable=rank != 0)]
    steps_per_epoch = sum(
        sum((count + args.batch_size - 1) // args.batch_size for count in
            (sum(not action for plan in cluster for _, action in plan),
             sum(action for plan in cluster for _, action in plan))) for cluster in plans)
    total_steps = args.epochs * steps_per_epoch
    total_samples = sum(len(plan) for cluster in plans for plan in cluster)
    total_action = sum(action for cluster in plans for plan in cluster for _, action in plan)
    if rank == 0:
        print(json.dumps({"event": "start", "steps": total_steps, "epochs": args.epochs,
                          "samples_per_epoch": total_samples, "mask_only_per_epoch": total_samples - total_action,
                          "action_chunks_per_epoch": total_action, "batch_size": args.batch_size,
                          "rank_batch_sizes": args.rank_batch_sizes, "world_size": world,
                          "device_name": torch.cuda.get_device_name(), "source_tcow_step": source_step,
                          "trainable_tcow": sum(p.numel() for p in model.tcow.parameters() if p.requires_grad),
                          "trainable_flow": sum(p.numel() for p in model.flow.parameters()),
                          "fresh_flow": False, "flow_type": model.flow.flow_type}), flush=True)
    step, best, wait_s = 0, -1.0, 0.0
    history, checked_phases = [], set()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    shared_root = Path("/dev/shm") / args.output.name
    if rank == 0:
        shared_root.mkdir(mode=0o700)
    with ThreadPoolExecutor(max_workers=1) as loader:
        future = (loader.submit(cache_cluster, args.data, clusters[0], shared_root / "0", args.flow_only)
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
                                        shared_root / str(cluster_index + 1), args.flow_only)
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
                    samples = [training_clip(episodes[i][1], end, action, "cuda")
                               for i, end, action in batch_items]
                    rgbd, query, truth, proprio, action, valid = (torch.cat(parts) for parts in zip(*samples))
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
                        logits, velocity, latent, context = engine(rgbd, query, proprio, noisy, tau)
                        action_loss = (((velocity.float() - target).square() * valid[:, :, None]).sum()
                                       * (world * bool(count)) / (valid_count * 6)) if is_action else logits.new_zeros(())
                    mask_loss = (logits.sum() * 0 if args.flow_only else
                                 original_mask_loss(losses, logits, truth, step / max(total_steps, 1))
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
                        if rank == 0:
                            print(json.dumps({"event": "gradient_smoke", "action_phase": is_action,
                                              "latent_shape": list(latent.shape),
                                              "context_shape": list(context.shape) if context is not None else None,
                                              "patch_grad": values[0], "mask_head_grad": values[1],
                                              "flow_grad": values[2]}), flush=True)
                        checked_phases.add(is_action)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["train_args"].gradient_clip,
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    logs = torch.stack((action_loss.detach(), mask_loss.detach()))
                    dist.all_reduce(logs)
                    logs /= world
                    progress.update()
                    progress.set_postfix(action=f"{logs[0].item():.3f}", mask=f"{logs[1].item():.3f}",
                                         chunks=len(batch) if is_action else 0, refresh=False)
                    # Release RGB-D clips and full-video masks before allocating the next batch.
                    del samples, rgbd, query, truth, proprio, action, valid
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
                if (cluster_index + 1) % args.eval_every_clusters == 0 or cluster_index + 1 == len(clusters) * args.epochs:
                    dist.barrier()
                    metrics = evaluate(model, args.data, val_rows, "cuda", distributed=True)
                    if rank == 0:
                        if args.closed_loop_every_evals and (len(history) + 1) % args.closed_loop_every_evals == 0:
                            metrics["closed_loop"] = evaluate_closed_loop(
                                model, args.data, val_rows, args.output / "validation" / f"step_{step}", step)
                        if best < 0 or metrics["action_mae"] < best:
                            best = metrics["action_mae"]
                            save_checkpoint(args.output / "best.pt", model, step, args, metrics)
                        record = {"step": step, "clusters_done": cluster_index + 1,
                                  "epoch": cluster_index // len(clusters) + 1,
                                  "validation": metrics, "best_action_mae": best,
                                  "action_loss": float(logs[0]), "mask_loss": float(logs[1]),
                                  "elapsed_s": time.perf_counter() - started, "prefetch_wait_s": wait_s,
                                  "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30,
                                  "peak_cpu_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
                        history.append(record)
                        tqdm.write(json.dumps(record), file=sys.stdout)
                    dist.barrier()
    if rank == 0:
        shared_root.rmdir()
        save_checkpoint(args.output / "final.pt", model, step, args, history[-1]["validation"])
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps({"event": "complete", "steps": step,
                          "elapsed_s": time.perf_counter() - started}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
