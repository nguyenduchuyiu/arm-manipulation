"""Cache frozen current-view ViT tokens before the trainable visual adapter."""
import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import torch
from tqdm.auto import tqdm

from OATFlow.policy.loader import LeRobotVideoFrames
from OATFlow.prior.model import PriorPolicy
from OATFlow.prior.train import PriorDataset, episode_arrays


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 2**20), b""):
            result.update(block)
    return result.hexdigest()


def sources(data, weights):
    return {key: digest(path) for key, path in dict(manifest=data / "manifest.jsonl",
            vision=weights).items()}


# A combinatorial minibatch spans up to 96 episodes.
@lru_cache(maxsize=128)
def shard(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def copy_cached_frames(output, cached, starts):
    previous = cached["starts"].numpy()
    if not len(previous) or np.any(np.diff(previous) <= 0):
        raise ValueError("reuse cache starts must be nonempty, sorted and unique")
    positions = np.searchsorted(previous, starts)
    reused = (positions < len(previous)) & (previous[np.minimum(positions, len(previous) - 1)] == starts)
    for key in output:
        if cached[key].shape != (len(previous), 64, 768) or cached[key].dtype != output[key].dtype:
            raise ValueError("reuse cache token shape/precision differs")
        output[key][reused] = cached[key][positions[reused]]
    return reused


def retire_cached_shard(previous, replacement):
    """Delete temporary old tokens only after verifying their replacement."""
    old = torch.load(previous, map_location="cpu", weights_only=True, mmap=True)
    new = torch.load(replacement, map_location="cpu", weights_only=True, mmap=True)
    positions = torch.searchsorted(new["starts"], old["starts"])
    if (positions >= len(new["starts"])).any() or not torch.equal(new["starts"][positions], old["starts"]):
        raise ValueError("replacement must retain every old cached frame")
    for key in ("overview", "wrist"):
        if not torch.equal(new[key][positions], old[key]):
            raise ValueError("replacement changed reused feature values")
    del old, new
    shard.cache_clear()
    previous.unlink()


class CachedPriorDataset(PriorDataset):
    def __getitem__(self, index):
        episode, chunk, _active = self.samples[index]
        row = self.rows[episode]
        arrays = episode_arrays(self.root / row["path"], self.horizon)
        cached = shard(self.videos[row["path"]])
        if not np.array_equal(cached["starts"].numpy(), arrays["starts"]):
            raise ValueError("cached feature/action frame mismatch")
        return (cached["overview"][chunk].float(), cached["wrist"][chunk].float(),
                torch.from_numpy(arrays["joint_anchor"][chunk]),
                torch.tensor(arrays["meta"]["task_id"]), torch.tensor(arrays["meta"]["target_id"]),
                torch.from_numpy(arrays["action"][chunk]), torch.from_numpy(arrays["valid"][chunk]))

    def __getitems__(self, indices):
        return [self[index] for index in indices]


def load_cache(directory, data, weights, rows):
    index = json.loads((directory / "index.json").read_text())
    if index["status"] != "complete" or any(index["sources"].get(key) != value
            for key, value in sources(data, weights).items()):
        raise ValueError("incomplete/stale ViT cache")
    files = {row["path"]: directory / row["file"] for row in index["episodes"]}
    if set(files) != {row["path"] for row in rows} or not all(p.is_file() for p in files.values()):
        raise ValueError("ViT cache must cover exactly the train episodes")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "vision-weights", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--ram-gib", type=int, default=10,
                        help="token bytes in /dev/shm; remaining shards on disk")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32",
                        help="token storage precision; float16 trades small rounding error for half the space")
    parser.add_argument("--reuse-cache", type=Path,
                        help="reuse matching frames from a complete cache of the same data/vision weights")
    parser.add_argument("--retire-reuse-cache", action="store_true",
                        help="verify each new shard, then delete its old temporary shard to reclaim space")
    args = parser.parse_args()
    if args.retire_reuse_cache and args.reuse_cache is None:
        parser.error("--retire-reuse-cache requires --reuse-cache")
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in (args.data / "manifest.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["split"] == "train" and r["group"] == "standard"]
    starts_by_path = {r["path"]: episode_arrays(args.data / r["path"])["starts"]
                      for r in tqdm(rows, desc="index expert frames", file=sys.stdout, mininterval=5)}
    episode_arrays.cache_clear()
    storage_dtype = getattr(torch, args.dtype)
    element_size = torch.empty((), dtype=storage_dtype).element_size()
    reused_files = {}
    if args.reuse_cache is not None:
        reused_config = json.loads((args.reuse_cache / "config.json").read_text())
        if reused_config["dtype"] != args.dtype:
            raise ValueError("reuse cache storage precision differs")
        reused_files = load_cache(args.reuse_cache, args.data, args.vision_weights, rows)
    size = sum(len(starts) for starts in starts_by_path.values()) * 2 * 64 * 768 * element_size
    if args.ram_gib < 0:
        raise ValueError("nonnegative RAM cache budget required")
    ram_budget = min(args.ram_gib * 2**30, size)
    reclaimable = 0
    if args.retire_reuse_cache:
        old_ram = Path(reused_config["ram_directory"])
        if any(path.parent not in (args.reuse_cache, old_ram) or path.stat().st_nlink != 1
               for path in reused_files.values()):
            raise ValueError("only dedicated old feature shards can be retired")
        device = args.output.parent.stat().st_dev
        reclaimable = sum(path.stat().st_size for path in reused_files.values() if path.stat().st_dev == device)
    if shutil.disk_usage("/dev/shm").free < ram_budget + 4 * 2**30:
        raise MemoryError("RAM token cache needs its budget plus 4GiB for DataLoader IPC")
    if shutil.disk_usage(args.output.parent).free + reclaimable < size - ram_budget + 6 * 2**30:
        raise MemoryError("ViT cache needs token bytes plus 6GiB for checkpoint and tests")
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high")
    model = PriorPolicy(args.vision_weights).cuda().eval()
    args.output.mkdir(parents=True)
    ram_directory = Path(tempfile.mkdtemp(prefix="oatflow_prior_", dir="/dev/shm"))
    config = dict(data=str(args.data), stride=1, vision_weights=str(args.vision_weights),
                  sources=sources(args.data, args.vision_weights), dtype=args.dtype,
                  precision="lossless encoder outputs" if args.dtype == "float32" else "rounded to float16 for storage; restored to float32 before adapters",
                  ram_directory=str(ram_directory), ram_budget_bytes=ram_budget,
                  tokens_per_view=64, feature_dim=768, estimated_bytes=size,
                  reuse_cache=str(args.reuse_cache) if args.reuse_cache is not None else None,
                  retire_reuse_cache=args.retire_reuse_cache)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    readme = "# Frozen ViT cache\n\nCurrent-view ViT tokens before trainable adapters. " + json.dumps(config) + "\nTraining/test not run here. Status: caching.\n"
    (args.output / "README.md").write_text(readme)
    records, ram_used, reused_frames, encoded_frames = [], 0, 0, 0
    if args.retire_reuse_cache:
        previous_index = json.loads((args.reuse_cache / "index.json").read_text())
        previous_index.update(status="retiring", replacement=str(args.output))
        (args.reuse_cache / "index.json").write_text(json.dumps(previous_index, indent=2) + "\n")
    for index, row in enumerate(tqdm(rows, desc="cache frozen ViT", file=sys.stdout, mininterval=5)):
        path = args.data / row["path"]
        starts = starts_by_path[row["path"]]
        output = {key: torch.empty((len(starts), 64, 768), dtype=storage_dtype)
                  for key in ("overview", "wrist")}
        reused = np.zeros(len(starts), bool)
        if reused_files:
            reused = copy_cached_frames(output, shard(reused_files[row["path"]]), starts)
        missing = np.flatnonzero(~reused)
        reused_frames += int(reused.sum()); encoded_frames += len(missing)
        views = [LeRobotVideoFrames(str(path / name), row["frames"], (320, 320))[starts[missing]]
                 for name in ("rgb.mp4", "wrist_rgb.mp4")] if len(missing) else []
        for begin in range(0, len(missing), 64):
            images = [torch.from_numpy(view[begin:begin + 64].copy()).permute(0, 3, 1, 2).cuda() for view in views]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                features = model.encode_views(*images)
            for key, value in zip(output, features):
                output[key][missing[begin:begin + 64]] = value.to(dtype=storage_dtype, device="cpu")
        name = f"episode_{index:04d}.pt"
        tensors = dict(starts=torch.from_numpy(starts), **output)
        byte_count = sum(value.numel() * value.element_size() for value in tensors.values())
        destination = ram_directory / name if ram_used + byte_count < ram_budget else args.output / name
        torch.save(tensors, destination)
        if destination.parent == ram_directory:
            ram_used += destination.stat().st_size
        records.append(dict(path=row["path"], file=str(destination), chunks=len(starts)))
        if args.retire_reuse_cache:
            retire_cached_shard(reused_files[row["path"]], destination)
            if index % 32 == 31:
                (args.output / "index.json").write_text(json.dumps(dict(status="caching", sources=config["sources"],
                    episodes=records, reused_frames=reused_frames, encoded_frames=encoded_frames), indent=2) + "\n")
    (args.output / "index.json").write_text(json.dumps(dict(status="complete", sources=config["sources"], episodes=records,
                        reused_frames=reused_frames, encoded_frames=encoded_frames), indent=2) + "\n")
    (args.output / "README.md").write_text(readme.replace("Status: caching", "Status: complete") +
                        f"\nReused {reused_frames} frame features; encoded {encoded_frames} missing frames.\n")
    if args.retire_reuse_cache:
        previous_index["status"] = "cleaned"
        (args.reuse_cache / "index.json").write_text(json.dumps(previous_index, indent=2) + "\n")
        with (args.reuse_cache / "README.md").open("a") as file:
            file.write(f"\nTemporary tensors retired after verifying every reused frame in {args.output}.\n")
    print(json.dumps(dict(event="vision_cache_complete", reused_frames=reused_frames,
                          encoded_frames=encoded_frames)), flush=True)


if __name__ == "__main__":
    main()
