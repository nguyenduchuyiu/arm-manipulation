"""Collect each expert with its extra demos, audit, then finalize actions and normalization."""
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
from OATFlow.dataset.augment_dataset import collect_scene, PROFILES, SETTINGS, MAX_TRIALS, MIN_DEMOS
from OATFlow.dataset.audit_tcow_dataset import audit
from OATFlow.dataset.episode import generate
from OATFlow.dataset.references import create_references
from OATFlow.dataset.validate import validate
from OATFlow.policy.data import expand_demonstrations


def _generate_one(job):
    root, output, seed, target, swaps, layout, tcow_labels, group, extra_demos, action_mode = job
    result = generate(output, seed, target, swaps, previews=False, layout=layout,
                      tcow_labels=tcow_labels)
    parent = output / f"episode_{result['seed']:06d}_{result['target_object_id']}"
    row = dict(path=str(parent.relative_to(root)), seed=result["seed"],
               split=result["split"], group=group)
    extra = None
    if extra_demos and group == "standard" and result["split"] == "train":
        validate(parent)
        extra = collect_scene((root, row, root / "augmentation"), extra_demos, action_mode)
    return result, row, extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, help="frozen scene/split plan; collect all queries then audit")
    parser.add_argument("--action-mode", choices=("absolute", "delta", "ee"), default="absolute")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed-file", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("standard", "new_layout"), default="standard")
    parser.add_argument("--tcow-labels", action="store_true")
    parser.add_argument("--extra-demos", type=int, choices=(0, 2, 3), default=0,
                        help="attach 2–3 successful perturbed demos per train sample (delta or ee mode)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.episodes < 1:
        raise ValueError("use 1–8 workers and positive episodes")
    if args.extra_demos:
        if args.action_mode not in ("delta", "ee") or args.layout != "standard":
            raise ValueError("extra demos require delta/ee actions and standard layout")
        args.tcow_labels = True
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
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / "collection.json").write_text(json.dumps({**config, "visual_input": "rgb"}, indent=2) + "\n")
    description = (f"# {args.output.name}\n\nGenerate synchronized 25 Hz physical expert episodes.\n"
                   f"Action mode: {args.action_mode}; wrist RGB + overview RGB (no depth); TCOW labels: {args.tcow_labels}.\n"
                   f"Workers: {args.workers}; requested extra train demos: {args.extra_demos}.\n"
                   "Training/weights/epochs/batch/LR/inference: not run.\n"
                   "Test: episode alignment, scene/split audit, fixed-anchor joint/EE H25 at stride10.\n"
                   "Extra demos use continuous actuator perturbation, executed-action labels, independent histories.\n"
                   "Each worker collects an expert and its extras together; normalization runs once after all demos.\n"
                   "Failed rollouts excluded; validation/test receive no extra demos; normalize all valid train demos.\n"
                   "Config: collection.json; dataset.json; augmentation/config.json when enabled. Status: collecting.\n")
    (args.output / "README.md").write_text(description)
    if args.plan:
        (args.output / "plan.jsonl").write_text(plan_text)
    if args.extra_demos:
        augmentation = args.output / "augmentation"
        augmentation.mkdir()
        for name in ("scenes", "scene_logs", "failed"):
            (augmentation / name).mkdir()
        config = dict(data=str(args.output), action_mode=args.action_mode, minimum_extra=MIN_DEMOS,
                      maximum_extra=args.extra_demos, maximum_trials=MAX_TRIALS,
                      settings=SETTINGS, profiles=PROFILES, workers=args.workers,
                      schedule="mild/medium/strong, then alternating mild/medium",
                      horizon=25, stride=10, fps=25,
                      collection="expert_then_extras_per_worker", normalization="after_all_demonstrations")
        (augmentation / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (augmentation / "README.md").write_text(description)
    jobs = []
    for scene in plan:
        output = args.output / scene["group"]
        create_references(output)
        jobs.extend((args.output, output, scene["seed"], target, scene["swaps"], scene["layout"],
                     args.tcow_labels, scene["group"] or "standard", args.extra_demos, args.action_mode)
                    for target in TARGETS)
    if not args.plan and not args.seed_file:
        jobs = jobs[:args.episodes]
    print(json.dumps({"event": "collection_start", "episodes": len(jobs), "workers": args.workers,
                      "action_mode": args.action_mode, "output": str(args.output)}), flush=True)
    rows = []
    extras = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(_generate_one, job): job for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="collect episodes",
                           unit="episode", mininterval=5, file=sys.stdout):
            result, row, extra = future.result()
            output = futures[future][1]
            with (output / "index.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            rows.append(row)
            if extra is not None:
                extras.append(extra)
                augmented = dict(status="running", completed=len(extras),
                                 added=sum(item["added"] for item in extras),
                                 shortfalls=[item["path"] for item in extras if item["status"] == "shortfall"])
                (augmentation / "summary.json").write_text(json.dumps(augmented, indent=2) + "\n")
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
    all_rows = rows
    summary = dict(episodes=len(rows), extra_demonstrations=0, status="complete")
    if args.extra_demos:
        if not extras:
            raise ValueError("no standard/train samples for extra demonstrations")
        augmented.update(status="shortfall" if augmented["shortfalls"] else "complete",
                         samples=len(extras), base_demonstrations=len(extras),
                         failed_trials=sum(item["failed_trials"] for item in extras))
        if augmented["shortfalls"]:
            (augmentation / "summary.json").write_text(json.dumps(augmented, indent=2) + "\n")
            raise RuntimeError(f"{len(augmented['shortfalls'])} samples have fewer than two extra demos")
        expanded = expand_demonstrations(args.output, [row for row in rows
                                       if row["group"] == "standard" and row["split"] == "train"])
        all_rows = expanded + [row for row in rows if row["group"] != "standard" or row["split"] != "train"]
        augmented["demonstrations_per_epoch"] = len(expanded)
        (augmentation / "summary.json").write_text(json.dumps(augmented, indent=2) + "\n")
        (augmentation / "README.md").write_text(description.replace("Status: collecting", "Status: complete") +
                                                "\n" + json.dumps(augmented, indent=2) + "\n")
        summary.update(extra_demonstrations=augmented["added"],
                       train_demonstrations_per_epoch=augmented["demonstrations_per_epoch"])
    finalize_actions(args.output, all_rows, args.action_mode)
    (args.output / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "README.md").write_text(description.replace("Status: collecting", "Status: complete") +
                                         "\n" + json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "generation_complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
