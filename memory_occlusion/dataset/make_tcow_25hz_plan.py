"""Freeze scene-disjoint train, validation, test, and composition splits."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from memory_occlusion.environment.env import TARGETS


HELD_OUT = (("Butter", "cover_b", 3), ("Tuna", "cover_a", 1))
PILOT_SCENES = 8
QUOTAS = {"train": {1: 54, 2: 53, 3: 53}, "val": {1: 7, 2: 7, 3: 6},
          "test": {1: 7, 2: 7, 3: 6}}


def scene(seed):
    objects = np.random.default_rng(seed).permutation(TARGETS)
    assignment = {str(name): "cover_a" if slot < 2 else "cover_b"
                  for slot, name in enumerate(objects)}
    swaps = int(np.random.default_rng(seed + 10000).integers(1, 4))
    held_out = next((name for name, cover, count in HELD_OUT
                     if assignment[name] == cover and swaps == count), None)
    return assignment, swaps, held_out


def expert_succeeds(seed, swaps):
    from memory_occlusion.dataset.episode import context, execute, make_context_plan
    from memory_occlusion.environment.env import MemoryOcclusionEnv

    plan = make_context_plan(seed, swaps)
    for target in TARGETS:
        with MemoryOcclusionEnv(context_fps=25, resolution=320) as env:
            env.reset(seed=seed, options={"upright_targets": True, "layout": "standard"})
            context(env, plan, lambda phase, valid: None)
            try:
                execute(env, target, lambda stage: None)
            except RuntimeError:
                return False
    return True


def _check(candidate):
    seed, _, swaps, _ = candidate
    return expert_succeeds(seed, swaps)


def candidates(split, composition):
    for seed in range(10000):
        if (split == "train" and seed % 10 >= 8 or
                split == "val" and seed % 10 != 8 or
                split == "test" and seed % 10 != 9):
            continue
        assignment, swaps, held_out = scene(seed)
        if (held_out is not None) == composition:
            yield seed, assignment, swaps, held_out


def select(split, group, quotas, pool):
    remaining = dict(quotas)
    chosen = []
    rejected = 0
    source = iter(candidates(split, group == "composition"))
    with tqdm(total=sum(quotas.values()), desc=f"{group}/{split} preflight",
              unit="scene", disable=pool is None, mininterval=5) as progress:
        while any(remaining.values()):
            batch = []
            while len(batch) < 16:
                candidate = next(source, None)
                if candidate is None:
                    break
                key = candidate[3] if group == "composition" else candidate[2]
                if remaining[key]:
                    batch.append(candidate)
            if not batch:
                raise RuntimeError(f"could not fill {group}/{split} quotas")
            verdicts = [True] * len(batch) if pool is None else list(pool.map(_check, batch))
            for (seed, assignment, swaps, held_out), good in zip(batch, verdicts):
                key = held_out if group == "composition" else swaps
                if not remaining[key]:
                    continue
                if not good:
                    rejected += 1
                    continue
                chosen.append({"seed": seed, "group": group, "split": split,
                               "layout": "standard", "swaps": swaps,
                               "assignment": assignment, "composition_case": held_out})
                remaining[key] -= 1
                progress.update(1)
            progress.set_postfix(rejected=rejected)
    return chosen, rejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preflight", action="store_true",
                        help="run the expert on all four queries before accepting a scene")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    rejected = 0
    executor = (ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn"))
                if args.preflight else nullcontext(None))
    with executor as pool:
        for split, quotas in QUOTAS.items():
            chosen, failed = select(split, "standard", quotas, pool)
            rows.extend(chosen)
            rejected += failed
        chosen, failed = select("test", "composition", {"Butter": 10, "Tuna": 10}, pool)
        rows.extend(chosen)
        rejected += failed
    pilot = []
    for swaps, count in ((1, 3), (2, 3), (3, 2)):
        pilot.extend([row for row in rows
                      if row["split"] == "train" and row["swaps"] == swaps][:count])
    if len(pilot) != PILOT_SCENES:
        raise ValueError("pilot does not cover all swap counts")
    pilot_seeds = {row["seed"] for row in pilot}
    rows = pilot + [row for row in rows if row["seed"] not in pilot_seeds]
    seeds = [row["seed"] for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("scene seed leaked between splits")
    args.output.mkdir(parents=True)
    (args.output / "plan.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for group in ("standard", "composition"):
        (args.output / f"{group}_seeds.txt").write_text(
            "".join(f"{row['seed']}\n" for row in rows if row["group"] == group))
    (args.output / "pilot_seeds.txt").write_text(
        "".join(f"{row['seed']}\n" for row in rows[:PILOT_SCENES]))
    summary = {"scene_seeds": len(rows), "episodes": 4 * len(rows),
               "expert_preflight": args.preflight, "rejected_scene_seeds": rejected,
               "by_split": dict(Counter(f"{row['group']}/{row['split']}" for row in rows)),
               "by_swaps": dict(Counter(str(row["swaps"]) for row in rows)),
               "held_out": [list(case) for case in HELD_OUT]}
    (args.output / "plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
