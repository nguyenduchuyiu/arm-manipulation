"""Balanced physical demos starting with both covers closed, without TCOW labels."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
import shutil
import sys

import numpy as np
from tqdm.auto import tqdm

from controllers.oracle_pick import GraspFailure, IKFailure
from OATFlow.dataset.actions import finalize_actions
from OATFlow.environment.env import COVER_DROP_ZONE_XY, COVER_DROP_ZONE_HALF_SIZE, COVER_DROP_GOALS_XY, TARGETS
from OATFlow.task import TASKS
from OATFlow.dataset.episode import generate as generate_episode
from OATFlow.dataset.references import create_references


def generate(directory, task_id, demo_id, seed):
    return generate_episode(directory.parent.parent, seed, TASKS[task_id]["objects"][demo_id % 2],
                            include_context=False, task_id=task_id, demo_id=demo_id,
                            directory=directory)


def collect_one(job):
    root, task_id, demo_id, seed = job
    directory = root / f"task_{task_id:02d}" / f"demo_{demo_id:03d}"
    failures = []
    for attempt in range(20):
        actual_seed = seed + attempt * 100000
        try:
            meta = generate(directory, task_id, demo_id, actual_seed)
        except (GraspFailure, IKFailure) as error:
            failures.append(dict(seed=actual_seed, reason=str(error)))
            shutil.rmtree(directory)
            continue
        return dict(path=str(directory.relative_to(root)), task_id=task_id, target_id=meta["target_id"], demo_id=demo_id,
                    seed=actual_seed, split="train", group="standard", rejected_attempts=failures,
                    frames=meta["frames"])
    raise RuntimeError(f"task {task_id} demo {demo_id}: 20 failed expert scenes: {failures}")


def collect(output, demos_per_task=50, workers=4, start_seed=16000):
    if output.exists():
        raise FileExistsError(output)
    if demos_per_task < 1 or not 1 <= workers <= 4:
        raise ValueError("positive demos/task; 1–4 workers")
    output.mkdir(parents=True)
    config = dict(mode="expert", start_seed=start_seed, tasks=TASKS, demos_per_task=demos_per_task, workers=workers,
                  expert_profile="fast", objective="cover_drop_then_target_lift",
                  action_representation="absolute_joint", horizon=25, stride=1,
                  gripper="absolute", anchor="observed joints are proprioception only",
                  views=["overview", "wrist"], tokens_per_view=64, initial_state="both covers closed",
                  task_semantics="object pair + left/right cover; separate 4-way target embedding; drop cover then lift one target",
                  cover_drop_zone_xy=list(COVER_DROP_ZONE_XY), cover_drop_zone_half_size=list(COVER_DROP_ZONE_HALF_SIZE),
                  cover_drop_goals_xy=COVER_DROP_GOALS_XY, cover_release_height_m=.12, target_lift_height_m=.12)
    (output / "collection.json").write_text(json.dumps(config, indent=2) + "\n")
    description = (f"# {output.name}\n\nOracle manipulation: 12 pair/cover tasks, {demos_per_task} demos/task.\n"
                   "Both covers closed initially; random cover XY, object XY/yaw, arm pose and grasp offset.\n"
                   "RGB overview/wrist, 25 Hz, H25 normalized absolute joint setpoints + absolute gripper.\n"
                   "Fast expert always; cover released from12cm in broad nearby green zone, then bilateral target lift ends task.\n"
                   "No cover lowering/home or target transport/placement. No TCOW labels/history.\n"
                   "Train weights/epochs/batch/LR: not run here. Test: physical cover drop/target lift and absolute command checks.\n"
                   "Config: collection.json, dataset.json, normalization.json. Status: collecting.\n")
    (output / "README.md").write_text(description)
    create_references(output)
    jobs = [(output, task_id, demo_id, start_seed + task_id * 1000 + demo_id)
            for demo_id in range(demos_per_task) for task_id in range(12)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(collect_one, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(jobs), desc="oracle demos", unit="demo", file=sys.stdout):
            row = future.result()
            rows.append(row)
            with (output / "manifest.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            tqdm.write(json.dumps(dict(event="demo_complete", **row)))
    counts = np.bincount([row["task_id"] for row in rows], minlength=12)
    if not np.all(counts == demos_per_task) or len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("unbalanced task coverage or repeated scene seed")
    if demos_per_task % 2 == 0:
        for task in TASKS:
            for target in task["objects"]:
                count = sum(row["task_id"] == task["task_id"] and row["target_id"] == TARGETS.index(target) for row in rows)
                if count != demos_per_task // 2:
                    raise ValueError("unbalanced target coverage within task")
    finalize_actions(output, rows)
    summary = dict(status="complete", demos=len(rows), per_task=counts.tolist(),
                   rejected_attempts=sum(len(row["rejected_attempts"]) for row in rows))
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "README.md").write_text(description.replace("Status: collecting", "Status: complete"))
    print(json.dumps(dict(event="collection_complete", **summary)), flush=True)
