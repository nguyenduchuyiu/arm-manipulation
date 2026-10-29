"""Home-start pick with four free XY positions rather than four fixed spawn regions."""

from sam3_pick.paths import PROJECT, ASSETS, OUTPUTS, DATASETS
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import numpy as np
import mujoco
from tqdm import tqdm
from sam3_pick.simulation.scene import (
    SEEN,
    UNSEEN,
    PERMS,
    NAMES,
    Scene,
    split,
    write_json,
)
from sam3_pick.data.recording import save
from sam3_pick.data.expert_recording import capture
from sam3_pick.data.pick_setup import setup, HOME
from sam3_pick.simulation.expert import execute
from sam3_pick.simulation.grasp import GraspFailure, IKFailure
from sam3_pick.utils.hashing import sha256_file as digest

ROOT = OUTPUTS / "data_collection"
SOURCE = ASSETS / "pick_scene.xml"
DATA = DATASETS / "pick_wide_rgbd"
LOW = np.array([0.3, -0.42])
HIGH = np.array([0.78, -0.1])
SEED = 1170000


def scene_spec(seed):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(LOW, HIGH, (4, 2))
    if (
        min((np.linalg.norm(a - b) for i, a in enumerate(positions) for b in positions[i + 1 :]))
        < 0.055
    ):
        raise ValueError("overlapping free XY points")
    positions = positions[np.lexsort((positions[:, 1], positions[:, 0]))]
    return dict(
        seed=int(seed),
        centers=positions.tolist(),
        positions=positions.tolist(),
        yaw=rng.uniform(-np.pi, np.pi, 4).tolist(),
        initial_arm=[0.0] * 5,
    )


def collect(job):
    root, g, pilot = job
    directory = Path(root) / f"g{g:03d}"
    if (directory / "rows.json").exists():
        rows = json.loads((directory / "rows.json").read_text())
        spec = json.loads((directory / "layout.json").read_text())
        return (rows, spec, spec["rejected"])
    model = mujoco.MjModel.from_xml_path(str(Path(root) / "scene.xml"))
    scene = Scene(model)
    rejected = []
    permutations = [0, 1] if pilot else SEEN if 32 <= g < 36 else range(24)
    staging = Path(root) / f".g{g:03d}_partial"
    if staging.exists():
        shutil.rmtree(staging)
    for attempt in range(1000):
        seed = SEED + g + attempt * 1000
        staging.mkdir()
        try:
            spec = scene_spec(seed)
            spec.pop("initial_arm")
            rows = []
            for p in tqdm(permutations, desc=f"Wide expert g{g:03d}", mininterval=10, leave=False):
                for t, target in enumerate(NAMES[:4]):
                    initial = setup(scene, spec, p)
                    store = {}
                    expert = execute(
                        scene, target, lambda phase: capture(scene, store, phase, True)
                    )
                    capture(scene, store, "done", False)
                    path = staging / f"p{p:02d}" / f"t{t}"
                    frames = save(path, store, initial=initial)
                    dataset_split, evaluation_group = split(g, p)
                    row = dict(
                        path=f"g{g:03d}/p{p:02d}/t{t}",
                        layout_id=g,
                        layout_seed=seed,
                        permutation=p,
                        final_permutation=p,
                        target_id=t,
                        target_object_id=target,
                        target_slot=PERMS[p].index(t),
                        slot_object_ids=PERMS[p],
                        split=dataset_split,
                        evaluation_group=evaluation_group,
                        frames=frames,
                        fps=25,
                        resolution=[320, 240],
                        views=["front", "wrist"],
                        depth_views=["front"],
                        objective="target_lift",
                        success=True,
                        initial_robot="settled common home",
                        context_frames=0,
                        decision_frames=dict(t_obj=0),
                        action_representation="absolute_joint_command_physical_rad_m",
                        **expert,
                    )
                    write_json(path / "episode.json", row)
                    rows.append(row)
                write_json(
                    Path(root) / f"progress_g{g:03d}.json",
                    dict(
                        permutations=len(rows) // 4,
                        total_permutations=len(permutations),
                        attempt=attempt,
                        episodes=len(rows),
                    ),
                )
            write_json(staging / "layout.json", dict(**spec, rejected=rejected))
            write_json(staging / "rows.json", rows)
            staging.rename(directory)
            return (rows, spec, rejected)
        except (GraspFailure, IKFailure, ValueError) as error:
            rejected.append(dict(seed=seed, error=str(error)))
            shutil.rmtree(staging)
            write_json(Path(root) / f"rejections_g{g:03d}.json", rejected)
    raise RuntimeError(f"No complete feasible layout g{g}: {rejected[-1]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = args.output
    sources = [
        Path(__file__).resolve(),
        Path(__file__).with_name("pick_setup.py").resolve(),
        PROJECT / "src/sam3_pick/simulation/expert.py",
        PROJECT / "src/sam3_pick/simulation/grasp.py",
        SOURCE,
        PROJECT / "src/sam3_pick/simulation/scene.py",
        PROJECT / "src/sam3_pick/data/recording.py",
        PROJECT / "src/sam3_pick/data/expert_recording.py",
        HOME,
    ]
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(SOURCE, out / "scene.xml")
        shutil.copyfile(OUTPUTS / "notes/e117_README.md", out / "README.md")
        config = dict(
            status="collecting_physics",
            pilot=args.pilot,
            seed=SEED,
            expected_layouts=2 if args.pilot else 40,
            expected_episodes=16 if args.pilot else 3712,
            train_layouts=list(range(32)),
            val_layouts=list(range(32, 36)),
            test_layouts=list(range(36, 40)),
            train_permutations=SEEN,
            unseen_permutations=UNSEEN,
            views=["front", "wrist"],
            depth_views=["front"],
            resolution=[320, 240],
            fps=25,
            depth_unit_m=0.002,
            depth_max_m=6.5535,
            camera_schema="pick_wide_front_rgbd_wrist_fisheye_rgb_v1",
            wrist_diagonal_fov_deg=175,
            visualization_rotation_deg=[0, 180],
            sample_hz=25,
            physics_hz=500,
            initial_robot="settled common home",
            home=str(HOME),
            xy_low_m=LOW.tolist(),
            xy_high_m=HIGH.tolist(),
            min_center_distance_m=0.055,
            sampling="four uniform XY points, min-separation rejection, sorted by X; no fixed spawn centers",
            feasibility="reject whole layout on any physical expert failure; record rejected seeds",
            split="new XY draws from same range; independent geometry and color-permutation holdouts",
            action_representation="absolute_joint_command_physical_rad_m",
            source_sha256={str(p): digest(p) for p in sources},
        )
        config["source_sha256"][str(out / "scene.xml")] = digest(out / "scene.xml")
        write_json(out / "config.json", config)
    config = json.loads((out / "config.json").read_text())
    assert config["pilot"] == args.pilot
    for p, expected in config["source_sha256"].items():
        assert digest(Path(p)) == expected, p
    groups = [0, 36] if args.pilot else list(range(40))
    rows = []
    failures = []
    start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(groups)), mp_context=mp.get_context("spawn")
    ) as pool:
        futures = [pool.submit(collect, (str(out), g, args.pilot)) for g in groups]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Free XY pick layouts",
            mininterval=5,
        ):
            current, spec, rejected = future.result()
            rows += current
            failures += rejected
            write_json(
                out / "progress.json",
                dict(
                    layouts=len({r["layout_id"] for r in rows}),
                    episodes=len(rows),
                    rejected_layouts=len(failures),
                    elapsed_s=time.perf_counter() - start,
                ),
            )
    rows.sort(key=lambda row: row["path"])
    assert len(rows) == config["expected_episodes"]
    if not args.pilot:
        assert Counter((r["evaluation_group"] for r in rows)) == dict(
            train=2048, val=256, composition=1024, spatial=256, combined=128
        )
    train = [r for r in rows if r["split"] == "train"]
    assert {r["layout_id"] for r in train}.isdisjoint(
        {r["layout_id"] for r in rows if r["evaluation_group"] in ("val", "spatial", "combined")}
    )
    assert {r["permutation"] for r in train}.isdisjoint(
        {r["permutation"] for r in rows if r["evaluation_group"] in ("composition", "combined")}
    )
    (out / "manifest.jsonl").write_text("".join((json.dumps(r) + "\n" for r in rows)))
    write_json(
        out / "audit.json",
        dict(
            status="physics_passed",
            episodes=len(rows),
            groups=dict(Counter((r["evaluation_group"] for r in rows))),
            rejected=failures,
            all_experts_succeeded=all(
                (r["success"] and r["max_bilateral_hold_frames"] >= 10 for r in rows)
            ),
        ),
    )
    config.update(status="physics_complete", physics_elapsed_s=time.perf_counter() - start)
    write_json(out / "config.json", config)


if __name__ == "__main__":
    main()
