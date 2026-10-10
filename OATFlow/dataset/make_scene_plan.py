"""Plan independent context+expert demos using the balanced prior scene setup."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from OATFlow.environment.env import TARGETS
from OATFlow.task import TASKS, task_id_from_state


HELD_OUT = ((TARGETS[0], "cover_b", 3), (TARGETS[3], "cover_a", 1))
QUOTAS = {"train": 70, "val": 10, "test": 10}


def assignment(task_id, swaps):
    initial = TASKS[task_id ^ (swaps % 2)]
    cover = ("cover_a", "cover_b")[initial["cover_side"]]
    return {name: cover if name in initial["objects"] else
            next(other for other in ("cover_a", "cover_b") if other != cover)
            for name in TARGETS}


def held_out(task_id, swaps):
    owners = assignment(task_id, swaps)
    return next((name for name, cover, count in HELD_OUT
                 if owners[name] == cover and swaps == count), None)


def standard_swaps(task_id, demo_id):
    allowed = [count for count in (1, 2, 3) if held_out(task_id, count) is None]
    if allowed == [2]:
        return 2
    if allowed == [1, 2]:
        return 1
    if allowed == [2, 3]:
        return 3
    if allowed != [1, 2, 3]:
        raise ValueError("unexpected composition constraint")
    # Alternate query pairs so neither target gets a fixed shuffle count.
    return (1, 3)[(demo_id // 2 + task_id) % 2]


def entries():
    rows = []
    for split, count in QUOTAS.items():
        base, suffix = {"train": (40000, None), "val": (60000, 8), "test": (80000, 9)}[split]
        for demo_id in range(count):
            for task in TASKS:
                task_id = task["task_id"]
                seed = base + task_id * 1000 + demo_id * 10 + (demo_id % 8 if suffix is None else suffix)
                swaps = standard_swaps(task_id, demo_id)
                rows.append(dict(seed=seed, group="standard", split=split, layout="standard",
                                 task_id=task_id, demo_id=demo_id,
                                 target=task["objects"][demo_id % 2], swaps=swaps,
                                 assignment=assignment(task_id, swaps), composition_case=None))
    for case_id, (case, _cover, swaps) in enumerate(HELD_OUT):
        for demo_id in range(60):
            seed = 100000 + case_id * 10000 + demo_id * 10 + 9
            target = TARGETS[(demo_id + demo_id // 4 + 2 * case_id) % 4]
            allowed = [task["task_id"] for task in TASKS if target in task["objects"]
                       and held_out(task["task_id"], swaps) == case]
            task_id = int(np.random.default_rng(seed).choice(allowed))
            rows.append(dict(seed=seed, group="composition", split="test", layout="standard",
                             task_id=task_id, demo_id=demo_id, target=target, swaps=swaps,
                             assignment=assignment(task_id, swaps), composition_case=case))
    return rows


def expert_succeeds(row):
    import mujoco
    from controllers.oracle_pick import GraspFailure, IKFailure
    from OATFlow.dataset.episode import context, setup_context_scene
    from OATFlow.dataset.expert import execute
    from OATFlow.environment.env import MemoryOcclusionEnv

    with MemoryOcclusionEnv(resolution=320) as env:
        try:
            rng, plan = setup_context_scene(env, row["task_id"], row["seed"], row["swaps"])
            geoms = [env.model.geom(name + "_visual").id for name in TARGETS]
            frames = [0]
            def record_context(phase, valid):
                if phase in ("shuffle", "hold") and np.isin(env.segmentation()[:, :, 0], geoms).any():
                    raise GraspFailure("object visible during shuffle/hold")
                frames[0] += 1
            context(env, plan, record_context)
            if env.assignment != row["assignment"]:
                raise ValueError("prior setup differs from planned identities")
            if task_id_from_state(env, TASKS[row["task_id"]]["objects"]) != row["task_id"]:
                raise ValueError("post-shuffle task differs from planned side")
            def record(stage):
                if stage != "t_obj":
                    mujoco.mj_forward(env.model, env.data)
                    frames[0] += 1
            execute(env, row["target"], rng, record)
        except (GraspFailure, IKFailure):
            return False
    return True


def checked_entry(row):
    rejected = []
    for attempt in range(20):
        candidate = {**row, "seed": row["seed"] + attempt * 1000000}
        if expert_succeeds(candidate):
            return {**candidate, "rejected_seeds": rejected}
        rejected.append(candidate["seed"])
    raise RuntimeError(f"20 failed seeds for task{row['task_id']} demo{row['demo_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = entries()
    if args.preflight:
        with ProcessPoolExecutor(max_workers=4, mp_context=get_context("spawn")) as pool:
            rows = list(tqdm(pool.map(checked_entry, rows), total=len(rows),
                             desc="prior context preflight", unit="demo", mininterval=5))
    if len(rows) != 1200 or len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("expected1200 independent demo seeds")
    for split, count in QUOTAS.items():
        counts = Counter(row["task_id"] for row in rows if row["group"] == "standard" and row["split"] == split)
        if counts != Counter({task_id: count for task_id in range(12)}):
            raise ValueError("unbalanced standard task counts")
    args.output.mkdir(parents=True)
    (args.output / "plan.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for group in ("standard", "composition"):
        (args.output / f"{group}_seeds.txt").write_text("".join(f"{r['seed']}\n" for r in rows if r["group"] == group))
    (args.output / "pilot_seeds.txt").write_text("".join(f"{r['seed']}\n" for r in rows[:8]))
    summary = dict(scene_seeds=len(rows), episodes=len(rows), queries_per_seed=1,
                   setup="shared prior setup_scene", expert_preflight=args.preflight,
                   rejected_scene_seeds=sum(len(row.get("rejected_seeds", [])) for row in rows),
                   by_split=dict(Counter(f"{r['group']}/{r['split']}" for r in rows)),
                   by_target=dict(Counter(r["target"] for r in rows)),
                   by_swaps=dict(Counter(str(r["swaps"]) for r in rows)),
                   held_out=[list(case) for case in HELD_OUT])
    (args.output / "plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
