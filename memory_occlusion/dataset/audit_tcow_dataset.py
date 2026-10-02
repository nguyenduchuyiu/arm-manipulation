"""Validate a 25 Hz TCOW dataset and write its episode manifest."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import sys
import json
from pathlib import Path
from tqdm.auto import tqdm

from memory_occlusion.dataset.validate import validate
from memory_occlusion.environment.env import TARGETS


def audit(root: Path, pilot_scenes=0, workers=1):
    plan = [json.loads(line) for line in (root / "plan.jsonl").read_text().splitlines()]
    if pilot_scenes:
        plan = plan[:pilot_scenes]
    if not plan or len({row["seed"] for row in plan}) != len(plan):
        raise ValueError("empty plan or scene seed appears in multiple splits")
    rows = []
    contexts = {}
    for scene in tqdm(plan, desc="audit scenes", unit="scene", mininterval=5):
        group, seed, split = scene["group"], scene["seed"], scene["split"]
        for target in TARGETS:
            path = root / group / f"episode_{seed:06d}_{target}"
            if not (path / "episode.json").is_file():
                raise FileNotFoundError(path / "episode.json")
            meta = json.loads((path / "episode.json").read_text())
            if (meta["seed"], meta["target_object_id"], meta["layout"], meta["split"],
                    meta["swaps"], meta["object_to_occluder"]) != (
                    seed, target, scene["layout"], split, scene["swaps"], scene["assignment"]):
                raise ValueError(f"episode metadata differs from frozen plan: {path}")
            if "context_plan" not in meta:
                raise ValueError(f"missing randomized context plan: {path}")
            if seed in contexts and contexts[seed] != meta["context_plan"]:
                raise ValueError(f"target queries use different contexts: {path}")
            contexts[seed] = meta["context_plan"]
            rows.append({"path": str(path.relative_to(root)), "seed": seed,
                         "target": target, "layout": scene["layout"], "group": group,
                         "split": split, "swaps": meta["swaps"],
                         "cover": meta["correct_cover_body"],
                         "composition_case": scene["composition_case"],
                         "is_held_out_query": group == "composition" and target == scene["composition_case"],
                         "frames": meta["frames"], "fps": meta["fps"],
                         "resolution": meta["resolution"],
                         "camera_fovy": meta["overview_camera"]["fovy"],
                         "cover_released_frame": meta["semantic_transition_frames"]["cover_released"]})
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        for _ in tqdm(pool.map(validate, [root / row["path"] for row in rows]),
                      total=len(rows), desc="audit episodes", unit="episode", mininterval=5, file=sys.stdout):
            pass
    if any(row["resolution"] != [240, 320] or row["camera_fovy"] != 50
           for row in rows):
        raise ValueError("dataset mixes camera settings or resolutions")
    summary = {"episodes": len(rows), "scene_seeds": len(plan),
               "by_group_split": dict(Counter(f"{row['group']}/{row['split']}"
                                              for row in rows)),
               "by_target": dict(Counter(row["target"] for row in rows)),
               "by_swaps": dict(Counter(str(row["swaps"]) for row in rows)),
               "by_cover": dict(Counter(row["cover"] for row in rows))}
    summary["held_out_queries"] = sum(row["is_held_out_query"] for row in rows)
    paths = [swap for context in contexts.values() for swap in context["swaps"]]
    summary["trajectory_diversity"] = {
        "unique_context_plans": len({json.dumps(context, sort_keys=True)
                                     for context in contexts.values()}),
        "swap_frames_range": [min(path["frames"] for path in paths),
                              max(path["frames"] for path in paths)],
        "arc_m_range": [min(path["arc_m"] for path in paths),
                        max(path["arc_m"] for path in paths)],
        "route_flip_counts": dict(Counter(str(path["route_flip"]) for path in paths)),
    }
    prefix = "pilot_" if pilot_scenes else ""
    (root / f"{prefix}manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    (root / f"{prefix}audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--pilot-scenes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("use 1–8 audit workers")
    audit(args.root, args.pilot_scenes, args.workers)


if __name__ == "__main__":
    main()
