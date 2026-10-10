"""Train the prior policy on balanced demos using current-frame RGB only."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from OATFlow.dataset.actions import chunk_starts
from OATFlow.prior.model import PriorPolicy
from OATFlow.policy.loader import LeRobotVideoFrames, cache_videos, make_dataloader, release_videos, training_batches


@lru_cache(maxsize=128)
def episode_arrays(path, horizon=25):
    meta = json.loads((path / "episode.json").read_text())
    with np.load(path / "supervision.npz") as z:
        action, valid = z["expert_action"], z["action_valid"]
    with np.load(path / "observation.npz") as z:
        state = z["joint_position"]
    if horizon < 1:
        raise ValueError("positive action horizon required")
    starts, stop = chunk_starts(valid, meta["decision_frames"]["t_occ"], path.name)
    indices = starts[:, None] + np.arange(horizon)
    return dict(starts=starts, action=action[np.minimum(indices, stop - 1)].copy(),
                valid=indices < stop, joint_anchor=state[starts].copy(), meta=meta)


class PriorDataset(Dataset):
    def __init__(self, root, rows, samples, videos, horizon=25):
        self.root, self.rows, self.samples, self.videos = root, rows, samples, videos
        self.horizon = horizon

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        items = [self.samples[index] for index in indices]
        decoded = {}
        for episode in {item[0] for item in items}:
            path = self.root / self.rows[episode]["path"]
            arrays = episode_arrays(path, self.horizon)
            chunks = sorted({item[1] for item in items if item[0] == episode})
            frames = arrays["starts"][chunks]
            views = [LeRobotVideoFrames(self.videos[(path / name).resolve()], arrays["meta"]["frames"], (320, 320))[frames]
                     for name in ("rgb.mp4", "wrist_rgb.mp4")]
            decoded[episode] = arrays, {chunk: index for index, chunk in enumerate(chunks)}, views
        result = []
        for episode, chunk, _active in items:
            arrays, positions, views = decoded[episode]
            view_index = positions[chunk]
            result.append((*(torch.from_numpy(view[view_index].copy()).permute(2, 0, 1) for view in views),
                           torch.from_numpy(arrays["joint_anchor"][chunk]),
                           torch.tensor(arrays["meta"]["task_id"], dtype=torch.long),
                           torch.tensor(arrays["meta"]["target_id"], dtype=torch.long),
                           torch.from_numpy(arrays["action"][chunk]),
                           torch.from_numpy(arrays["valid"][chunk])))
        return result


def restore_training_state(model, optimizer, saved):
    if saved["architecture"] != f"oracle_task_target_{model.action_head_type}":
        raise ValueError("checkpoint action head differs from model")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    for parameter, state in optimizer.state.items():
        if state["exp_avg"].shape != parameter.shape or state["exp_avg_sq"].shape != parameter.shape:
            raise ValueError("optimizer moments differ from model parameters")
    return dict(epoch=saved["epoch"], step=saved["step"], optimizer_states=len(optimizer.state),
                optimizer_step_min=min(float(state["step"]) for state in optimizer.state.values()),
                optimizer_step_max=max(float(state["step"]) for state in optimizer.state.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--vision-cache", type=Path, help="frozen ViT token cache from OATFlow.prior.features")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vision-weights", type=Path, required=True)
    parser.add_argument("--flow-weights", type=Path, required=True)
    parser.add_argument("--action-head", choices=('fm', 'act'), default='fm')
    parser.add_argument("--resume", type=Path, help="restore policy and optimizer; --epochs counts additional epochs")
    parser.add_argument("--restart-scheduler", action="store_true",
                        help="with --resume, restart warmup/cosine for additional epochs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=25, choices=(25, 50))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=3e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.restart_scheduler and args.resume is None:
        raise ValueError("scheduler restart requires --resume")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if min(args.epochs, args.batch_size) < 1 or args.lr <= 0:
        raise ValueError("positive epochs, batch and learning rate required")
    if not 0 < args.min_lr <= args.lr or args.warmup_steps < 1:
        raise ValueError("positive min LR <= peak LR; positive warmup")
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads((args.data / "generation_summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError("collection must be complete before training")
    rows = [json.loads(line) for line in (args.data / "manifest.jsonl").read_text().splitlines()]
    if len(rows) != summary.get("demos", summary.get("episodes")):
        raise ValueError("manifest differs from collection summary")
    if "per_task" in summary and np.bincount([r["task_id"] for r in rows], minlength=12).tolist() != summary["per_task"]:
        raise ValueError("manifest task coverage differs from collection summary")
    rows = [row for row in rows if row["split"] == "train" and row["group"] == "standard"]
    if not rows:
        raise ValueError("no standard/train expert chunks")
    statistics = json.loads((args.data / "normalization.json").read_text())
    if statistics["action_representation"] != "absolute_joint":
        raise ValueError("absolute joint data required")
    representation, action_file = "absolute_joint", "supervision.npz"
    for path in (args.vision_weights, args.flow_weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    resumed = None
    start_epoch, source_step = 0, 0
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        if "scheduler" not in resumed:
            raise ValueError("resume requires a checkpoint with saved LR scheduler")
        source_config = resumed["config"]
        if source_config.get('action_head', 'fm') != args.action_head:
            raise ValueError('resume action head differs')
        fixed = ("batch_size", "seed", "horizon") if args.restart_scheduler else ("batch_size", "lr", "min_lr", "warmup_steps", "seed", "horizon")
        if Path(source_config["data"]).resolve() != args.data.resolve() or any(
                source_config[name] != getattr(args, name) for name in fixed):
            raise ValueError("resume data, batch, seed or LR configuration differs")
        if source_config.get("stride") != 1:
            raise ValueError("resume requires stride1 training")
        start_epoch, source_step = resumed["epoch"], resumed["step"]
    collection = json.loads((args.data / "collection.json").read_text())
    cluster_size = 96 if collection.get("objective") == "target_lift" else 4
    rng = np.random.default_rng(args.seed)
    if cluster_size == 96:
        rows.sort(key=lambda row: (row["seed"], row["permutation"], row["target_id"]))
    else:
        rows = [rows[index] for index in rng.permutation(len(rows))]
    plans = []
    if cluster_size == 96:
        for start in range(0, len(rows), cluster_size):
            group = rows[start:start + cluster_size]
            if len(group) != 96 or len({r["seed"] for r in group}) != 1:
                raise ValueError("pick training requires complete contiguous permutation groups")
            if [(r["permutation"], r["target_id"]) for r in group] != [(p,t) for p in range(24) for t in range(4)]:
                raise ValueError("pick rows must be ordered by permutation and target")
    for start in tqdm(range(0, len(rows), cluster_size), desc="index prior chunks", file=sys.stdout):
        plans.append([[(index, True) for index in range(len(episode_arrays(
                          args.data / row["path"], args.horizon)["starts"]))]
                      for row in rows[start:start + cluster_size]])
    scheduler_start_step = source_step if args.restart_scheduler else source_config.get("scheduler_start_step", 0) if resumed is not None else 0
    scheduler_start_epoch = start_epoch if args.restart_scheduler else source_config.get("scheduler_start_epoch", 0) if resumed is not None else 0
    planning_epochs = start_epoch + args.epochs - scheduler_start_epoch
    batch_planner = training_batches
    if cluster_size == 96:
        from OATFlow.pick.batching import combinatorial_batches
        batch_planner = combinatorial_batches
    samples, batches, epoch_ends = batch_planner(plans, planning_epochs, args.batch_size, rng)
    completed_epochs = start_epoch - scheduler_start_epoch
    completed_steps = source_step - scheduler_start_step
    if completed_epochs:
        if completed_steps != epoch_ends[completed_epochs - 1]:
            raise ValueError("resume checkpoint must be at the matching epoch boundary")
    batches = batches[completed_steps:]
    epoch_ends = [scheduler_start_step + end for end in epoch_ends[completed_epochs:]]
    if resumed is not None and not args.restart_scheduler:
        if source_step + len(batches) < resumed["config"]["total_steps"]:
            raise ValueError("resume cannot shorten the original training target")
    schedule_steps = (len(batches) if args.restart_scheduler else
                      resumed["config"]["scheduler_steps"] if resumed is not None else len(batches))
    if args.warmup_steps >= schedule_steps:
        raise ValueError("warmup must end before the final training step")
    model = (PriorPolicy(horizon=args.horizon, action_head=args.action_head) if resumed is not None else
             PriorPolicy(args.vision_weights, args.flow_weights, horizon=args.horizon, action_head=args.action_head)).cuda()
    if resumed is not None and statistics != resumed["config"]["statistics"]:
        raise ValueError("resume normalization differs from source training")
    model.head.set_statistics(statistics)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.lr, betas=(.9, .95), weight_decay=1e-10)
    scheduler = CosineDecayWithWarmupSchedulerConfig(
        num_warmup_steps=args.warmup_steps, num_decay_steps=schedule_steps,
        peak_lr=args.lr, decay_lr=args.min_lr).build(optimizer, num_training_steps=source_step + len(batches))
    if resumed is not None:
        restoration = restore_training_state(model, optimizer, resumed)
        if args.restart_scheduler:
            # Restoring Adam overwrites its LR; put it back at the fresh schedule's first value.
            for group, lr, base_lr in zip(optimizer.param_groups, scheduler.get_last_lr(), scheduler.base_lrs):
                group["lr"] = lr
                group["initial_lr"] = base_lr
        else:
            scheduler.load_state_dict(resumed["scheduler"])
            if scheduler.last_epoch != source_step - scheduler_start_step:
                raise ValueError("scheduler step differs from checkpoint step")
        torch.set_rng_state(resumed["torch_rng_state"])
        torch.cuda.set_rng_state(resumed["cuda_rng_state"])
        model.initialization = {"resume": str(args.resume), "restored": restoration,
                                "original_flow_source": str(args.flow_weights),
                                "original_vision_source": str(args.vision_weights),
                                "scheduler_steps": schedule_steps,
                                "scheduler_restarted": args.restart_scheduler,
                                "restored_lr": optimizer.param_groups[0]["lr"],
                                "noise_rng": "restored CPU and CUDA RNG states"}
        print(json.dumps(dict(event="resume_loaded", **model.initialization)), flush=True)
        del resumed
    args.output.mkdir(parents=True)
    config = {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "architecture": f"oracle_task_target_{args.action_head}", "tcow": False, "context_tokens": 131,
              "overview_tokens": 64, "wrist_tokens": 64, "task_ids": 12, "target_ids": 4,
              "vision_encoder": "shared frozen ImageNet ViT-B/16; 320px; 20x20 -> 8x8 average pool",
              "trainable": f"visual adapter, view/task/target embeddings, proprio encoder, four decoder blocks, {args.action_head} head",
              "action_representation": representation, "action_dim": model.head.action_dim,
              "action_input_dim": model.head.action_dim, "action_output_dim": model.head.action_dim,
              "action_padding": False,
              "action_file": action_file, "horizon": args.horizon, "stride": 1,
              "flow_steps": 10 if args.action_head == 'fm' else 0,
              "context_activation_checkpointing": args.action_head == 'fm',
              "demos": len(rows), "chunks_per_epoch": len(samples),
              "start_epoch": start_epoch, "total_epochs": start_epoch + args.epochs,
              "source_step": source_step, "additional_steps": len(batches),
              "total_steps": source_step + len(batches), "statistics": statistics,
              "lr_scheduler": "cosine_decay_with_warmup",
              "scheduler_steps": schedule_steps,
              "scheduler_start_step": scheduler_start_step, "scheduler_start_epoch": scheduler_start_epoch,
              "validation": "not run; held-out closed-loop evaluation required"}
    if args.action_head == 'act':
        config['act'] = dict(width=512, decoder_layers=1, posterior_layers=4, latent_dim=32,
                             kl_weight=10., loss='masked normalized L1 +10*KL', inference='zero latent; single decoder pass')
    config["collection"] = collection
    config["batch_cluster_episodes"] = cluster_size
    config["batching"] = "all24 permutations per full batch, no chunk resampling; shuffled layout order" if cluster_size == 96 else "episode clusters"
    if collection.get("objective") == "target_lift":
        config["conditioning"] = "target identity and constant task0; no permutation/slot/layout ID"
    config["objective"] = config["collection"].get("objective", "unspecified")
    if args.vision_cache is not None:
        config["vision_cache_storage"] = json.loads((args.vision_cache / "config.json").read_text())
        config["vision_cache_open_shards_per_worker"] = 128
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "flow_initialization.json").write_text(json.dumps(model.initialization, indent=2) + "\n")
    (args.output / "README.md").write_text(
        f"# {args.output.name}\n\nTask+target conditioned {args.action_head.upper()} pretraining without TCOW.\n"
        f"Data: {args.data}; {len(rows)} demos. Weights: {args.vision_weights}, {args.flow_weights}.\n"
        f"Frozen ImageNet vision; trainable visual/proprio adapters, embeddings, decoder and {args.action_head.upper()}.\n"
        f"Epochs {args.epochs}; batch {args.batch_size}; LR {args.lr}; seed {args.seed}; 131 context tokens; head {args.action_head}.\n"
        f"LR schedule: linear warmup {args.warmup_steps} updates, peak {args.lr}, cosine decay to {args.min_lr}.\n"
        f"Scheduler horizon {schedule_steps} updates; restart {args.restart_scheduler}; schedule starts at global step {config['scheduler_start_step']}.\n"
        f"Continuation: {args.resume}; epochs {start_epoch + 1}–{start_epoch + args.epochs}; "
        f"{len(batches)} additional updates from step {source_step}, policy and optimizer restored when resuming.\n"
        f"Actions: H{args.horizon} {representation}, absolute gripper, 25Hz; inference {config.get('act', {}).get('inference', 'Euler10')}.\n"
        "Chunks start at every valid expert frame (stride1); short tails are masked.\n"
        f"Objective: {config['objective']}; collection/geometry settings in config.json.\n"
        f"Batching: {config['batching']}.\n"
        "Test: held-out simulation success not run. Config: config.json, flow_initialization.json. Status: training.\n")
    dataset_class = PriorDataset
    if args.vision_cache is not None:
        from OATFlow.prior.features import CachedPriorDataset, load_cache
        videos = load_cache(args.vision_cache, args.data, args.vision_weights, rows)
        dataset_class = CachedPriorDataset
    else:
        videos, video_gib = cache_videos(args.data, rows, True)
        (args.output / "video_cache.json").write_text(json.dumps(dict(directory=str(Path(next(iter(videos.values()))).parent), gib=video_gib), indent=2))
    loader = make_dataloader(dataset_class(args.data, rows, samples, videos, args.horizon), batches, args.seed, True)
    started = time.perf_counter()
    history = []
    loss_sum, loss_count = 0., 0
    metric_sums = dict(l1=0., kl=0.) if args.action_head == 'act' else {}
    torch.cuda.reset_peak_memory_stats()
    try:
        progress = tqdm(enumerate(loader, source_step + 1), total=len(batches), desc=f"train {args.action_head.upper()} policy", mininterval=5, file=sys.stdout)
        for step, tensors in progress:
            used_lr = optimizer.param_groups[0]["lr"]
            overview, wrist, proprio, task_id, target_id, action, valid = [tensor.cuda(non_blocking=True) for tensor in tensors]
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                metrics = {}
                if args.action_head == 'act':
                    if args.vision_cache is None:
                        overview, wrist = model.encode_views(overview, wrist)
                    (prediction, mu, logvar), context = model.act_forward_features(
                        overview, wrist, proprio, task_id, target_id, action, valid)
                    loss, metrics = model.act.loss(prediction, action, valid, mu, logvar)
                else:
                    noisy, time_fraction, truth = model.flow.training_path(action)
                    forward = model.forward_features if args.vision_cache is not None else model
                    velocity, context = forward(overview, wrist, proprio, task_id, target_id, noisy, time_fraction)
                    weights = valid[:, :, None]
                    loss = ((velocity.float() - truth).square() * weights).sum() / (weights.sum() * action.shape[-1])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite policy loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step == source_step + 1:
                gradients = {name: float(sum(p.grad.float().abs().sum() for p in module.parameters() if p.grad is not None))
                             for name, module in (("task", model.task_embedding), ("target", model.target_embedding),
                                                  ("decoder", model.context_decoder), (args.action_head, model.head),
                                                  ("vision_projection", model.visual_projection), ("proprio", model.head.proprio))}
                if any(not np.isfinite(value) or value <= 0 for value in gradients.values()) or any(p.grad is not None for p in model.vision_encoder.parameters()):
                    raise ValueError("missing trainable gradient or frozen vision gradient")
                print(json.dumps(dict(event="gradients_checked", gradients=gradients, context_shape=list(context.shape),
                                      loss=float(loss), lr=used_lr, peak_vram_gib=torch.cuda.max_memory_reserved() / 2**30)), flush=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss)
            for name, value in metrics.items():
                metric_sums[name] += float(value)
            loss_count += 1
            progress.set_postfix(loss=f"{float(loss):.4f}", lr=f"{used_lr:.2e}",
                                 **{name: f'{float(value):.4f}' for name,value in metrics.items()}, refresh=False)
            if step % 100 == 0:
                print(json.dumps(dict(event="train_progress", step=step, total_steps=source_step + len(batches),
                                      loss=float(loss), lr=used_lr, **{name:float(value) for name,value in metrics.items()})), flush=True)
            if step in epoch_ends:
                epoch = start_epoch + epoch_ends.index(step) + 1
                saved = dict(architecture=config['architecture'], model=model.state_dict(),
                             optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                             torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state(),
                             config=config, step=step, epoch=epoch)
                torch.save(saved, args.output / "latest.tmp")
                (args.output / "latest.tmp").replace(args.output / "latest.pt")
                record = dict(epoch=epoch, step=step, mean_loss=loss_sum / loss_count,
                              lr=used_lr, next_lr=optimizer.param_groups[0]["lr"],
                              elapsed_s=time.perf_counter() - started,
                              peak_vram_gib=torch.cuda.max_memory_reserved() / 2**30)
                record.update({f'mean_{name}': value/loss_count for name,value in metric_sums.items()})
                history.append(record)
                (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
                print(json.dumps(record), flush=True)
                loss_sum, loss_count = 0., 0
                metric_sums = {name:0. for name in metric_sums}
        (args.output / "final.pt").hardlink_to(args.output / "latest.pt")
        p = args.output / "README.md"
        p.write_text(p.read_text().replace("Status: training", "Status: complete") + f"\nFinal metrics: {json.dumps(history[-1])}\n")
        print(json.dumps(dict(event="training_complete", **history[-1])), flush=True)
    finally:
        del loader
        if args.vision_cache is None:
            release_videos(videos)


if __name__ == "__main__":
    main()
