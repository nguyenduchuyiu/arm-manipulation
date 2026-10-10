"""Collect context + expert or expert-only multiview absolute-joint demonstrations."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm

from OATFlow.environment.env import TARGETS
from OATFlow.dataset.actions import finalize_actions
from OATFlow.dataset.collect_expert import collect
from OATFlow.dataset.audit_tcow_dataset import audit
from OATFlow.dataset.episode import generate
from OATFlow.dataset.references import create_references
from OATFlow.dataset.validate import validate
from OATFlow.task import TASKS


def _generate_one(job):
    output, scene = job
    return generate(output, scene["seed"], scene["target"], scene["swaps"],
                    previews=False, layout=scene["layout"], include_context=True,
                    task_id=scene["task_id"], demo_id=scene["demo_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="independent prior-style demo/split plan")
    parser.add_argument("--mode", choices=("context", "expert"), default="context")
    parser.add_argument("--demos-per-task", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("standard", "new_layout"), default="standard")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.episodes < 1:
        raise ValueError("use 1–8 workers and positive episodes")
    if args.mode == "expert":
        if args.plan or args.seed_file or args.layout != "standard":
            raise ValueError("expert mode uses balanced covered scenes; no context plan/layout")
        collect(args.output, args.demos_per_task, args.workers, args.start_seed)
        return
    if args.layout != "standard":
        raise ValueError("prior-style context uses standard layout")
    if args.output.exists():
        raise FileExistsError(f"use a fresh dataset directory: {args.output}")
    if args.plan:
        if args.seed_file or args.start_seed or args.layout != "standard":
            raise ValueError("plan specifies scene seeds and layouts")
        plan_text = args.plan.read_text()
        plan = [json.loads(line) for line in plan_text.splitlines()]
        if not plan or len({scene["seed"] for scene in plan}) != len(plan):
            raise ValueError("empty plan or repeated scene seed")
    else:
        seeds = ([int(line) for line in args.seed_file.read_text().splitlines()]
                 if args.seed_file else list(range(args.start_seed, args.start_seed + args.episodes)))
        if len(seeds) != len(set(seeds)):
            raise ValueError("duplicate scene seed")
        plan = [dict(seed=seed, group="", layout=args.layout, task_id=index % 12,
                     demo_id=index // 12, target=TASKS[index % 12]["objects"][(index // 12) % 2],
                     swaps=int(np.random.default_rng(seed + 10000).integers(1, 4)))
                for index, seed in enumerate(seeds)]
    args.output.mkdir(parents=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(scene_seeds=len(plan), episodes=len(plan), queries_per_seed=1,
                  setup="shared prior setup_scene")
    (args.output / "collection.json").write_text(json.dumps({**config, "visual_input": "rgb", "objective": "cover_drop_then_target_lift", "action_representation": "absolute_joint"}, indent=2) + "\n")
    description = (f"# {args.output.name}\n\nGenerate synchronized 25 Hz physical expert episodes.\n"
                   "Mode: context + expert; absolute joint, full 320px RGB views and cropped 240x320 TCOW labels.\n"
                   f"Independent scenes: {len(plan)}; one target per seed. Shared prior setup_scene randomizes arm±0.06rad, cover XY±25mm, object placement/yaw and grasp offsets.\n"
                   "Shuffle alternates sides each round, preserves object/cover identities and in-box position, and hands the same state to the prior expert.\n"
                   f"Workers: {args.workers}.\n"
                   "Training/weights/epochs/batch/LR/inference: not run.\n"
                   "Test: episode alignment, scene/split audit, absolute joint H25 at stride1.\n"
                   "Context actions excluded from FM; normalization uses valid standard/train expert frames only.\n"
                   "Config: collection.json; dataset.json. Status: collecting.\n")
    (args.output / "README.md").write_text(description)
    if args.plan:
        (args.output / "plan.jsonl").write_text(plan_text)
    jobs = []
    for scene in plan:
        output = args.output / scene["group"]
        create_references(output)
        jobs.append((output, scene))
    print(json.dumps({"event": "collection_start", "episodes": len(jobs), "workers": args.workers,
                      "action_representation": "absolute_joint", "output": str(args.output)}), flush=True)
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
                             seed=result["seed"], split=result["split"], group=output.name if args.plan else "standard"))
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
    finalize_actions(args.output, rows)
    summary = dict(episodes=len(rows), status="complete", mode="context")
    (args.output / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "README.md").write_text(description.replace("Status: collecting", "Status: complete") +
                                         "\n" + json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "generation_complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
