"""Generate complete 25 Hz expert episodes; stop on an unsuccessful expert."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from memory_occlusion.environment.env import TARGETS
from memory_occlusion.dataset.episode import generate
from memory_occlusion.dataset.references import create_references


def _generate_one(job):
    output, seed, target, swaps, layout, tcow_labels = job
    return generate(output, seed, target, swaps, previews=False, layout=layout,
                    tcow_labels=tcow_labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed-file", type=Path,
                        help="one scene seed per line; generates all four target queries")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-failed", action="store_true",
                        help="record expert failures and continue generating other seeds")
    parser.add_argument("--output", type=Path, default=Path("outputs/memory_occlusion/dataset"))
    parser.add_argument("--layout", choices=("standard", "new_layout"), default="standard")
    parser.add_argument("--tcow-labels", action="store_true",
                        help="save 240x320 target, occluder, and container masks at 25 Hz")
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    create_references(args.output)
    index_path = args.output / "index.jsonl"
    existing = [json.loads(line) for line in index_path.read_text().splitlines()] if index_path.exists() else []
    if any(row["layout"] != args.layout for row in existing):
        raise ValueError("use a separate output directory for each layout type")
    if any("semantic_transition_frames" not in row for row in existing):
        raise ValueError("output contains old sparse masks; generate dense masks in a new output directory")
    if any(bool(row.get("tcow_labels")) != args.tcow_labels for row in existing):
        raise ValueError("use a separate output directory for TCOW labels")
    completed = {(row["seed"], row["target_object_id"], row["swaps"]) for row in existing}
    failed_path = args.output / "failed.jsonl"
    failures = [json.loads(line) for line in failed_path.read_text().splitlines()] if failed_path.exists() else []
    failed = {(row["seed"], row["target_object_id"], row["swaps"]) for row in failures}

    def record(result):
        seed = result["seed"]
        with index_path.open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        completed.add((seed, result["target_object_id"], result["swaps"]))

    jobs = []
    seeds = ([int(line) for line in args.seed_file.read_text().splitlines()]
             if args.seed_file else
             [args.start_seed + index for index in range((args.episodes + 3) // 4)])
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed file contains duplicate scene seeds")
    for index, (seed, target) in enumerate(
            (seed, target) for seed in seeds for target in TARGETS):
        if not args.seed_file and index >= args.episodes:
            break
        swaps = int(np.random.default_rng(seed + 10000).integers(1, 4))
        if (seed, target, swaps) in completed:
            continue
        if (seed, target, swaps) in failed and args.skip_failed:
            continue
        directory = args.output / f"episode_{seed:06d}_{target}"
        required = ("episode.json", "input.json", "rgb.mp4", "observation.npz", "supervision.npz")
        if args.tcow_labels:
            required += ("tcow_labels.npz", "query_mask.png")
        if all((directory / name).exists() for name in required):
            result = json.loads((directory / "episode.json").read_text())
            if result["swaps"] == swaps and result["success"]:
                record(result)
                continue
        jobs.append((args.output, seed, target, swaps, args.layout, args.tcow_labels))
    if args.workers < 1:
        raise ValueError("workers must be positive")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(_generate_one, job): job for job in jobs}
        for index, future in enumerate(tqdm(as_completed(futures), total=len(futures),
                                            desc="generate episodes", unit="episode", mininterval=5), 1):
            _, seed, target, swaps, _, _ = futures[future]
            try:
                result = future.result()
            except RuntimeError as error:
                if not args.skip_failed:
                    raise
                with failed_path.open("a") as stream:
                    stream.write(json.dumps({"seed": seed, "target_object_id": target,
                                             "swaps": swaps, "error": str(error)}) + "\n")
                tqdm.write(f"{index}/{len(jobs)}: seed={seed}, target={target}, failed: {error}")
                continue
            record(result)
            tqdm.write(f"{index}/{len(jobs)}: seed={seed}, target={target}, {result['split']}")


if __name__ == "__main__":
    main()
