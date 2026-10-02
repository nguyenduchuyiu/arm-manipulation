"""Collect synchronized 25 Hz multiview episodes, audit, then finalize actions."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm

from memory_occlusion.environment.env import TARGETS
from memory_occlusion.dataset.actions import finalize_actions
from memory_occlusion.dataset.audit_tcow_dataset import audit
from memory_occlusion.dataset.episode import generate
from memory_occlusion.dataset.references import create_references
from memory_occlusion.dataset.validate import validate


def _generate_one(job):
    output, seed, target, swaps, layout, tcow_labels = job
    return generate(output, seed, target, swaps, previews=False, layout=layout,
                    tcow_labels=tcow_labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="frozen scene/split plan; collect all queries then audit")
    parser.add_argument("--action-mode", choices=("absolute", "delta"), default="absolute")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("standard", "new_layout"), default="standard")
    parser.add_argument("--tcow-labels", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.episodes < 1:
        raise ValueError("use 1–8 workers and positive episodes")
    if args.output.exists():
        raise FileExistsError(f"use a fresh dataset directory: {args.output}")
    if args.plan:
        if args.seed_file or args.start_seed or args.layout != "standard":
            raise ValueError("plan specifies scene seeds and layouts")
        plan_text = args.plan.read_text()
        plan = [json.loads(line) for line in plan_text.splitlines()]
        if not plan or len({scene["seed"] for scene in plan}) != len(plan):
            raise ValueError("empty plan or repeated scene seed")
        args.tcow_labels = True
    else:
        seeds = ([int(line) for line in args.seed_file.read_text().splitlines()]
                 if args.seed_file else list(range(args.start_seed, args.start_seed + (args.episodes + 3) // 4)))
        if len(seeds) != len(set(seeds)):
            raise ValueError("duplicate scene seed")
        plan = [dict(seed=seed, group="", layout=args.layout,
                     swaps=int(np.random.default_rng(seed + 10000).integers(1, 4))) for seed in seeds]
    args.output.mkdir(parents=True)
    if args.plan:
        (args.output / "plan.jsonl").write_text(plan_text)
    jobs = []
    for scene in plan:
        output = args.output / scene["group"]
        create_references(output)
        jobs.extend((output, scene["seed"], target, scene["swaps"], scene["layout"], args.tcow_labels)
                    for target in TARGETS)
    if not args.plan and not args.seed_file:
        jobs = jobs[:args.episodes]
    print(json.dumps({"event": "collection_start", "episodes": len(jobs), "workers": args.workers,
                      "action_mode": args.action_mode, "output": str(args.output)}), flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(_generate_one, job): job for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="collect episodes",
                           unit="episode", mininterval=5, file=sys.stdout):
            result = future.result()
            output = futures[future][0]
            with (output / "index.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            rows.append(dict(path=str((output / f"episode_{result['seed']:06d}_{result['target_object_id']}").relative_to(args.output)),
                             split=result["split"], group=output.name if args.plan else "standard"))
            tqdm.write(f"completed seed={result['seed']} target={result['target_object_id']} frames={result['frames']}")
    print("Collection complete; auditing every episode.", flush=True)
    if args.plan:
        rows = audit(args.output, workers=args.workers)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            for _ in tqdm(pool.map(validate, [args.output / row["path"] for row in rows]),
                          total=len(rows), desc="audit episodes", unit="episode", file=sys.stdout):
                pass
        (args.output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    finalize_actions(args.output, rows, args.action_mode)


if __name__ == "__main__":
    main()
