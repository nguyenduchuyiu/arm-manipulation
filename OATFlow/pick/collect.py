"""Generate complete 24-permutation x four-query groups, split by layout seed."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys

import imageio.v3 as iio
import mujoco
import numpy as np
from tqdm.auto import tqdm

from controllers.oracle_pick import GraspFailure, IKFailure
from OATFlow.dataset.actions import finalize_actions
from OATFlow.environment.env import MemoryOcclusionEnv
from OATFlow.pick.scene import PERMUTATIONS, execute, permutation_spec, scene_spec, setup
from OATFlow.task import TARGETS, normalized


def write_episode(env, path, spec, permutation, target, split):
    spec = permutation_spec(spec, permutation)
    initial = setup(env, spec, permutation)
    path.mkdir(parents=True)
    writers = [iio.imopen(path / name, "w", plugin="pyav") for name in ("rgb.mp4", "wrist_rgb.mp4")]
    for writer in writers:
        writer.init_video_stream("libx264", fps=25, pixel_format="yuv420p")
    proprio, actions, phases, valid, qpos, qvel = [], [], [], [], [], []

    def record(phase, action_valid=True):
        for writer, image in zip(writers, (env._overview(), env.wrist_image())):
            writer.write(image, is_batch=False)
        proprio.append(normalized(env.data.qpos[env.robot_qpos_addresses], env.model.actuator_ctrlrange))
        actions.append(normalized(env.data.ctrl, env.model.actuator_ctrlrange))
        phases.append(phase)
        valid.append(action_valid)
        qpos.append(env.data.qpos.copy())
        qvel.append(env.data.qvel.copy())

    try:
        record("pick_start", False)
        expert = execute(env, target, record)
        record("done", False)
    finally:
        for writer in writers:
            writer.close()
    n = len(phases)
    np.savez_compressed(path / "observation.npz", joint_position=np.array(proprio, np.float32), timestamp_s=np.arange(n) / 25)
    np.savez_compressed(path / "supervision.npz", expert_action=np.array(actions, np.float32),
                        action_valid=np.array(valid), phase=np.array(phases))
    np.savez_compressed(path / "sim_state.npz", initial_qpos=initial[0], initial_qvel=initial[1],
                        initial_ctrl=initial[2], qpos=np.array(qpos), qvel=np.array(qvel))
    meta = dict(objective="target_lift", seed=spec["seed"], layout_seed=spec["seed"], scene=spec,
                permutation=permutation, slot_objects=PERMUTATIONS[permutation],
                target_slot=PERMUTATIONS[permutation].index(target),
                task_id=0, task_id_semantics="constant pick task; no slot information",
                target_id=TARGETS.index(target), target_object_id=target, selected_cover=None,
                frames=n, fps=25, resolution=[320, 320], split=split, layout="standard",
                decision_frames=dict(t_occ=1, t_obj=1), action_representation="absolute_joint",
                success=True, **expert)
    (path / "episode.json").write_text(json.dumps(meta, indent=2) + "\n")
    return dict(path=str(path.name), task_id=0, target_id=TARGETS.index(target), seed=spec["seed"],
                permutation=permutation, target_slot=meta["target_slot"], split=split,
                group="standard", frames=n)


def collect_group(root, group, split, base_seed):
    env = MemoryOcclusionEnv()
    rejections = []
    try:
        # Reject whole groups, never individual queries/permutations: every
        # accepted geometry retains all target identities at every slot.
        for attempt in range(40):
            seed = base_seed + group + attempt * 10000
            try:
                spec = scene_spec(seed)
            except ValueError as ex:
                rejections.append(dict(seed=seed, error=str(ex)))
                continue
            try:
                queries = [(p, target) for p in range(24) for target in TARGETS]
                for permutation, target in tqdm(queries, desc=f"layout {group} feasibility #{attempt + 1}",
                                                mininterval=15, file=sys.stdout, leave=False):
                    current = permutation_spec(spec, permutation)
                    setup(env, current, permutation)
                    execute(env, target, lambda _phase: None)
            except (GraspFailure, IKFailure, ValueError) as ex:
                rejections.append(dict(seed=seed, error=str(ex)))
                continue
            directory = root / f"g{group:03d}"
            rows = []
            try:
                for permutation, target in tqdm(queries, desc=f"layout {group} record",
                                                mininterval=15, file=sys.stdout, leave=False):
                    target_id = TARGETS.index(target)
                    # Within each permutation, all queries share exactly
                    # the same initial physical state.
                    row = write_episode(env, directory / f"p{permutation:02d}_t{target_id}",
                                        spec, permutation, target, split)
                    row["path"] = str(Path(directory.name) / row["path"])
                    rows.append(row)
                return rows, spec, rejections
            except (GraspFailure, IKFailure) as ex:
                rejections.append(dict(seed=seed, error=str(ex), stage="full collection"))
                shutil.rmtree(directory)
        raise RuntimeError(f"group {group}: no feasible layout in 40 attempts: {rejections}")
    finally:
        env.close()


def audit(rows, root=None):
    if root is not None:
        from controllers.nexarm_mujoco_backend import JOINT_NAMES, MUJOCO_JOINTS
        from OATFlow.environment.env import SCENE_XML
        model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        robot_addresses = [int(model.joint(MUJOCO_JOINTS[name]).qposadr[0]) for name in JOINT_NAMES]
    groups = {}
    for row in rows:
        groups.setdefault(row["seed"], []).append(row)
    for seed, group in groups.items():
        if len(group) != 96 or len({(r["permutation"], r["target_id"]) for r in group}) != 96:
            raise ValueError(f"incomplete permutation/query group {seed}")
        if len({r["split"] for r in group}) != 1:
            raise ValueError("layout leaked across splits")
        cells = Counter((r["target_id"], r["target_slot"]) for r in group)
        if len(cells) != 16 or set(cells.values()) != {6}:
            raise ValueError("target/slot imbalance")
        if root is not None:
            group_robot = None
            for permutation in range(24):
                queries = sorted((r for r in group if r["permutation"] == permutation), key=lambda r: r["target_id"])
                reference = None
                for row in queries:
                    path = root / row["path"]
                    meta = json.loads((path / "episode.json").read_text())
                    target = TARGETS[row["target_id"]]
                    if (meta["target_object_id"] != target or meta["task_id"] != 0
                            or meta["slot_objects"][row["target_slot"]] != target
                            or tuple(meta["slot_objects"]) != PERMUTATIONS[permutation]):
                        raise ValueError("query/identity/slot mapping differs")
                    with np.load(path / "sim_state.npz") as state:
                        initial = np.r_[state["initial_qpos"], state["initial_qvel"], state["initial_ctrl"]]
                        robot = state["initial_qpos"][robot_addresses]
                        for slot, name in enumerate(meta["slot_objects"]):
                            adr = int(model.joint(name + "_joint").qposadr[0])
                            if np.linalg.norm(state["initial_qpos"][adr:adr + 2] - meta["scene"]["positions"][slot]) > .002:
                                raise ValueError("physical object/slot mapping differs")
                    if group_robot is None:
                        group_robot = robot
                    elif not np.allclose(robot, group_robot, atol=1e-10, rtol=0):
                        raise ValueError("permutation-dependent initial proprioception")
                    if reference is None:
                        reference = initial
                    elif not np.array_equal(initial, reference):
                        raise ValueError("query-dependent physical initialization")
    return dict(layout_groups=len(groups), episodes=len(rows),
                target_slot_counts={split: {f"{t}:{s}": sum(r["split"] == split and r["target_id"] == t and r["target_slot"] == s for r in rows)
                                           for t in range(4) for s in range(4)} for split in ("train", "val", "test")},
                layout_split_overlap=0, permutations_per_group=24, queries_per_permutation=4,
                initial_target_slot_chance_without_vision=.25,
                counterfactual_state_check="verified" if root is not None else "not run")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-groups", type=int, default=32)
    parser.add_argument("--val-groups", type=int, default=4)
    parser.add_argument("--test-groups", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=550000)
    args = parser.parse_args()
    if min(args.train_groups, args.val_groups, args.test_groups) < 1 or not 1 <= args.workers <= 8:
        raise ValueError("positive split sizes and one to eight workers required")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    config = dict(objective="target_lift", **{k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                  permutations=24, targets=4, target_identity_order=list(TARGETS),
                  policy_inputs="overview, wrist, proprio, target identity; constant task0",
                  slots="group-level randomized centers; independently randomized object XY/yaw per permutation, shared by four queries",
                  random_initial_pose="independent RNG stream; query-independent",
                  group_rejection="whole group if any expert fails; all rejection seeds logged",
                  split="disjoint complete layout groups; no per-trajectory split", fps=25,
                  action_representation="absolute_joint", expert_profile="fast")
    (args.output / "collection.json").write_text(json.dumps(config, indent=2) + "\n")
    readme = (f"# {args.output.name}\n\nPick-only pretraining data, 24 permutations x4 queries per layout. "
              f"Train/val/test groups {args.train_groups}/{args.val_groups}/{args.test_groups}; seed {args.seed}. "
              "Absolute6 joints at25Hz, fast physical expert. "
              "No covers, shuffle context, placement or TCOW. Geometry/robot RNG streams independent; "
              "query does not alter initial state. Whole-group feasibility rejection is recorded. "
              "Training/test not run. Training chunks start at every valid expert frame (stride1). "
              "Configuration: collection.json. Status: collecting.\n")
    (args.output / "README.md").write_text(readme)
    splits = ["train"] * args.train_groups + ["val"] * args.val_groups + ["test"] * args.test_groups
    rows, specs, rejected = [], [], []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        jobs = [pool.submit(collect_group, args.output, i, split, args.seed) for i, split in enumerate(splits)]
        with (args.output / "manifest.jsonl").open("w") as manifest:
            for future in tqdm(as_completed(jobs), total=len(jobs), desc="pick layout groups", file=sys.stdout):
                group_rows, spec, failures = future.result()
                rows.extend(group_rows); specs.append(spec); rejected.extend(failures)
                for row in group_rows:
                    manifest.write(json.dumps(row) + "\n")
                manifest.flush()
                print(json.dumps(dict(event="group_complete", seed=spec["seed"], episodes=len(rows), rejected=len(rejected))), flush=True)
    rows.sort(key=lambda r: r["path"])
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = audit(rows, args.output)
    result.update(rejected_groups=rejected, scenes=specs)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    finalize_actions(args.output, rows)
    (args.output / "generation_summary.json").write_text(json.dumps(dict(status="complete", demos=len(rows), **result), indent=2) + "\n")
    (args.output / "README.md").write_text(readme.replace("Status: collecting", "Status: complete") + f"\n{len(rows)} expert successes; audit.json.\n")


if __name__ == "__main__":
    main()
