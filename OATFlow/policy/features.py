"""Cache frozen, query-conditioned TCOW and wrist features before trainable adapters."""
import argparse
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from OATFlow.policy.data import sample_index, split_rows
from OATFlow.policy.loader import ActionChunkDataset, cache_videos, make_dataloader, release_videos
from OATFlow.policy.model import WristVisionEncoder, load_training_tracker, set_tracker_input
from OATFlow.policy.tracking import Seeker


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_sources(data, weights, config_checkpoint, wrist_weights):
    return {name: file_sha256(path) for name, path in (
        ("manifest", data / "manifest.jsonl"), ("tcow", weights),
        ("tcow_config", config_checkpoint), ("wrist", wrist_weights))}


@lru_cache(maxsize=4)
def feature_shard(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


class FeatureDataset(Dataset):
    def __init__(self, cache, rows, samples):
        self.records = {row["path"]: row for row in cache["episodes"]}
        self.rows, self.samples = rows, samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode, frame, active = self.samples[index]
        if not active:
            raise ValueError("feature cache contains expert action starts only")
        shard = feature_shard(self.records[self.rows[episode]["path"]]["file"])
        position = int(torch.searchsorted(shard["frame"], frame))
        if position == len(shard["frame"]) or int(shard["frame"][position]) != frame:
            raise ValueError(f"missing cached frame {frame}")
        return tuple(shard[name][position] for name in
                     ("latent", "proprio", "action", "valid", "wrist"))


def load_feature_cache(path, args, rows):
    cache = json.loads((path / "index.json").read_text())
    if cache["status"] != "complete":
        raise ValueError("feature cache is incomplete")
    if cache["sources"] != cache_sources(args.data, args.weights, args.config_checkpoint, args.wrist_weights):
        raise ValueError("cache encoders or dataset differ from training sources")
    if {row["path"] for row in rows} != {row["path"] for row in cache["episodes"]}:
        raise ValueError("cache must contain exactly the standard/train episodes")
    for row in cache["episodes"]:
        if not Path(row["file"]).is_file():
            raise FileNotFoundError(row["file"])
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "weights", "config-checkpoint", "wrist-weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ram-gib", type=int, default=20,
                        help="maximum cache shard bytes in /dev/shm; remaining shards go to output")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    rows, _ = split_rows(args.data)
    samples, batches, episodes = [], [], []
    for episode, row in enumerate(tqdm(rows, desc="index feature cache", unit="episode")):
        starts = [end for end, active in sample_index(args.data / row["path"]) if active]
        first = len(samples)
        samples.extend((episode, frame, True) for frame in starts)
        batches.extend(list(range(start, min(start + args.batch_size, len(samples))))
                       for start in range(first, len(samples), args.batch_size))
        episodes.append(dict(path=row["path"], frames=starts))
    # BF16 pre-adapter features plus exact FP32 state/actions and valid masks.
    sizes = [len(row["frames"]) * ((300 + 64) * 768 * 2 + (6 + 25 * 6) * 4 + 25 + 8)
             for row in episodes]
    ram_budget = args.ram_gib * 2**30
    if shutil.disk_usage("/dev/shm").free < ram_budget + 4 * 2**30:
        raise MemoryError("RAM cache needs its budget plus4GiB for video/DataLoader IPC")
    if shutil.disk_usage(args.output.parent).free < max(0, sum(sizes) - ram_budget) + 5 * 2**30:
        raise OSError("disk cache needs feature bytes plus5GiB for checkpoints")
    args.output.mkdir(parents=True)
    memory = Path(tempfile.mkdtemp(prefix="oatflow_features_", dir="/dev/shm"))
    used = 0
    for episode, row in enumerate(episodes):
        root = memory if used + sizes[episode] <= ram_budget else args.output
        if root == memory:
            used += sizes[episode]
        row["file"] = str(root / f"episode_{episode:04d}.pt")
    config = dict(status="running", data=str(args.data), sources=cache_sources(
        args.data, args.weights, args.config_checkpoint, args.wrist_weights),
        weights=str(args.weights), config_checkpoint=str(args.config_checkpoint),
        wrist_weights=str(args.wrist_weights), dtype="bfloat16", tokens=[300, 64],
        sample_count=len(samples), episodes=episodes, ram_directory=str(memory),
        ram_gib=used / 2**30, raw_gib=sum(sizes) / 2**30,
        stride=1, history_frames=30,
        cache_stage="pre-adapter TCOW current tokens and pooled wrist tokens; query included")
    (args.output / "index.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "README.md").write_text(
        f"# Frozen feature cache\n\nTCOW {args.weights}, wrist {args.wrist_weights}; "
        f"data {args.data}, {len(rows)} standard/train episodes, {len(samples)} expert starts. "
        "BF16 pre-adapter features,300TCOW+64wrist tokens,768channels. "
        "Each clip uses first-frame query and30past/current RGB frames. "
        "FP32 absolute-joint state/action labels and terminal validity retained. "
        f"RAM shards: {memory}; disk shards: this directory. "
        "Temporary RAM shards must remain until training completes. Config: index.json. Status: running.\n")
    tcow_config = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    source = torch.load(args.weights, map_location="cpu", weights_only=False, mmap=True)
    tracker = set_tracker_input(Seeker(logging.getLogger("tcow"), **dict(
        tcow_config["seeker_args"], tracker_pretrained=False)))
    step = load_training_tracker(tracker, source)
    wrist = WristVisionEncoder(tracker.seeker.tracker_backbone)
    wrist.load_vit_base(args.wrist_weights)
    tracker.requires_grad_(False).cuda().eval()
    wrist.freeze()
    wrist.cuda()
    videos, _ = cache_videos(args.data, rows, True)
    loader = make_dataloader(ActionChunkDataset(args.data, rows, samples, videos, True, True),
                             batches, 0, True)
    offset, current, chunks = 0, 0, []
    try:
        with torch.no_grad():
            for tensors in tqdm(loader, desc="cache frozen features", unit="batch", mininterval=5,
                                file=sys.stdout):
                rgb, query, _, proprio, action, valid, image = tensors
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    # Same backbone inputs as QueryMaskTracker.forward; no mask head needed.
                    dense, _ = tracker.seeker.tracker_backbone(torch.cat(
                        (rgb.cuda(non_blocking=True).float(), query.cuda(non_blocking=True).float()), 1), None)
                    latent = dense[:, :, -1].flatten(2).transpose(1, 2).contiguous()
                    features = wrist(image.cuda(non_blocking=True))
                    features = F.adaptive_avg_pool2d(features.transpose(1, 2).reshape(
                        len(image), 768, 20, 20), (8, 8)).flatten(2).transpose(1, 2).contiguous()
                if not torch.isfinite(latent).all() or not torch.isfinite(features).all():
                    raise ValueError("nonfinite frozen features")
                chunks.append(dict(latent=latent.cpu().to(torch.bfloat16), wrist=features.cpu().to(torch.bfloat16),
                                   proprio=proprio, action=action, valid=valid))
                offset += len(rgb)
                del dense, latent, features
                if offset == sum(len(row["frames"]) for row in episodes[:current + 1]):
                    shard = {name: torch.cat([part[name] for part in chunks]) for name in chunks[0]}
                    shard["frame"] = torch.tensor(episodes[current]["frames"], dtype=torch.int64)
                    torch.save(shard, episodes[current]["file"])
                    chunks, current = [], current + 1
    finally:
        del loader
        release_videos(videos)
    if current != len(episodes) or offset != len(samples):
        raise ValueError("feature cache did not cover all expert starts")
    config.update(status="complete", source_tcow_step=step)
    (args.output / "index.json").write_text(json.dumps(config, indent=2) + "\n")
    readme = args.output / "README.md"
    readme.write_text(readme.read_text().replace("Status: running", "Status: complete"))
    print(json.dumps({"event": "cache_complete", "samples": offset, "source_tcow_step": step,
                      "raw_gib": config["raw_gib"], "ram_gib": config["ram_gib"]}), flush=True)


if __name__ == "__main__":
    main()
