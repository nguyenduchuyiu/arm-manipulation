"""Attach 2–3 successful, continuously perturbed demonstrations per TRAIN scene."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from tqdm.auto import tqdm

from OATFlow.dataset.actions import save_action_chunks
from OATFlow.dataset.episode import generate, make_context_plan
from OATFlow.dataset.validate import validate
from OATFlow.environment.env import TARGETS
from OATFlow.policy.data import expand_demonstrations, split_rows
from OATFlow.task import normalized


IDENTITY = ("seed", "split", "layout", "target_object_id", "context_plan", "object_to_occluder")
PROFILES = [[1., .7, .7, 2., .05], [2., 1.2, 1.2, 4., .10], [3., 1.8, 1.8, 6., .15]]
SETTINGS = dict(tau_s=.6, retry_scale=.2, max_attempts=3, labels="executed")
MAX_TRIALS = 12
MIN_DEMOS, MAX_DEMOS = 2, 3


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noise_config(meta, trial):
    # First explore all three amplitudes; subsequent trials use mild/medium noise.
    profile = trial if trial < 3 else (trial - 3) % 2
    return {**SETTINGS, "std_deg": PROFILES[profile],
            "seed": 100000 + meta["seed"] * 1000 + TARGETS.index(meta["target_object_id"]) * 100 + trial}


def audit_demo(parent, parent_path, path, metadata, action_mode):
    if any(metadata[key] != parent[key] for key in IDENTITY):
        raise ValueError(f"perturbed demo changed its parent scene: {path}")
    if not metadata["success"] or metadata["control_perturbation"]["labels"] != "executed":
        raise ValueError(f"cannot attach unsuccessful/non-executed labels: {path}")
    validate(path)
    decision = metadata["decision_frames"]["t_occ"]
    with np.load(parent_path / "debug_poses.npz") as base, np.load(path / "debug_poses.npz") as extra:
        if not np.array_equal(base["qpos"][:decision + 1], extra["qpos"][:decision + 1]):
            raise ValueError(f"context/initial physical state differs from parent: {path}")
    with np.load(path / "supervision.npz") as gt, np.load(path / "observation.npz") as obs, \
            np.load(path / "perturbation.npz") as trace:
        valid = gt["action_valid"]
        frames = trace["frames"]
        if not np.array_equal(frames, np.flatnonzero(valid)):
            raise ValueError(f"noise trace does not span every action: {path}")
        actual = np.stack([normalized(command, np.asarray(metadata["normalization_limits"]))
                           for command in trace["executed_ctrl"]])
        if not np.array_equal(gt["expert_action"][frames], actual):
            raise ValueError(f"training labels differ from actual executed controls: {path}")
        if not np.array_equal(trace["expert_ctrl"][:, 5], trace["executed_ctrl"][:, 5]):
            raise ValueError(f"perturbation changed gripper commands: {path}")
        if not np.array_equal(gt["expert_action"], gt["executed_action"]):
            raise ValueError(f"selected supervision is not executed: {path}")
        delta = trace["executed_ctrl"][:, :5] - trace["expert_ctrl"][:, :5]
        bound = 2 * np.deg2rad(metadata["control_perturbation"]["std_deg"])
        if np.any(np.abs(delta) > bound * trace["noise_scale"][:, None] + 1e-12):
            raise ValueError(f"applied noise exceeds configured bounds: {path}")
        if not np.all(np.any(delta != 0, axis=1)):
            raise ValueError(f"an action tick has no perturbation: {path}")
        kinematics = None
        if action_mode == "ee":
            from OATFlow.dataset.ee_actions import EEKinematics
            kinematics = EEKinematics()
        chunks = save_action_chunks(path, gt["expert_action"], obs["joint_position"], valid,
                                    metadata, action_mode, kinematics)
    return len(chunks["starts"])


def attach(parent, entries):
    # Readers only see validated, completely written demonstrations.
    write_json(parent / "demonstrations.json", {"version": 1, "demonstrations": entries})


def collect_scene(job, maximum=MAX_DEMOS):
    root, row, output = job
    representation = json.loads((root / "normalization.json").read_text())["action_representation"]
    action_mode = {"relative_joint": "delta", "relative_ee": "ee"}[representation]
    parent = root / row["path"]
    meta = json.loads((parent / "episode.json").read_text())
    state_path = output / "scenes" / (parent.name + ".json")
    state = (json.loads(state_path.read_text()) if state_path.exists() else
             dict(path=row["path"], status="collecting", demonstrations=[], trials=[]))
    if state["status"] in ("complete", "shortfall"):
        return state
    demos = parent / "demos"
    demos.mkdir(exist_ok=True)
    if not (demos / "references").exists():
        shutil.copytree(parent.parent / "references", demos / "references")
    manifest = parent / "demonstrations.json"
    existing = json.loads(manifest.read_text())["demonstrations"] if manifest.exists() else []
    # Reconcile an interruption between publishing a demo and updating scene status.
    entries = []
    completed_trials = {record["trial"] for record in state["trials"]}
    for entry in existing:
        path = parent / entry["path"]
        extra = json.loads((path / "episode.json").read_text())
        if extra.get("augmentation_run") != str(output):
            raise ValueError(f"parent has demos from another augmentation run: {parent}")
        entries.append(entry)
        trial = extra["augmentation_trial"]
        if trial not in completed_trials:
            state["trials"].append(dict(trial=trial, success=True, id=entry["id"],
                                        frames=extra["frames"], resumed=True))
            completed_trials.add(trial)
    state["demonstrations"] = entries
    with (output / "scene_logs" / (parent.name + ".log")).open("a", buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        for trial in range(MAX_TRIALS):
            if len(entries) == maximum:
                break
            if trial in completed_trials:
                continue
            if shutil.disk_usage(root).free < 5 * 1024**3:
                raise OSError("less than 5 GiB free for augmentation")
            work = demos / f".trial{trial:02d}"
            if work.exists():
                shutil.rmtree(work)  # An incomplete, unpublished trial from this run.
            work.mkdir()
            noise = noise_config(meta, trial)
            description = (f"# {output.name}: {parent.name}, trial {trial}\n\n"
                "Continuous physical oracle + correlated joint perturbation, 25 Hz from shuffle end.\n"
                "Labels: executed commands; clean oracle commands also retained. Gripper unchanged.\n"
                f"RGB/wrist/proprio (no depth), {action_mode} H25, stride10; parent dataset normalization.\n"
                "No FM training, weights or inference. Recovery: max3 attempts, noise scale0.2.\n"
                f"Noise: {json.dumps(noise)}\nStatus: collecting.\n")
            (work / "README.md").write_text(description)
            print(json.dumps(dict(event="trial_started", path=row["path"], trial=trial,
                                  attached=len(entries), noise=noise)), file=sys.__stdout__, flush=True)
            started = time.monotonic()
            result = generate(demos, meta["seed"], meta["target_object_id"], swaps=meta["swaps"],
                              previews=False, layout="standard", tcow_labels=True,
                              control_perturbation=noise, directory=work)
            result.update(augmentation_run=str(output), augmentation_trial=trial)
            write_json(work / "episode.json", result)
            record = dict(trial=trial, success=result["success"], frames=result["frames"], noise=noise,
                          elapsed_s=time.monotonic() - started,
                          failure_reason=result["control_perturbation"]["failure_reason"])
            if result["success"]:
                record["action_chunks"] = audit_demo(meta, parent, work, result, action_mode)
                demo_id = f"a{len(entries) + 1:02d}"
                destination = demos / demo_id
                if destination.exists():
                    # Only an unpublished completed trial from this same run may be recovered.
                    old = json.loads((destination / "episode.json").read_text())
                    if old.get("augmentation_run") != str(output) or old["augmentation_trial"] != trial:
                        raise FileExistsError(destination)
                    shutil.rmtree(destination)
                (work / "README.md").write_text(description.replace("Status: collecting", "Status: success; validated and attached"))
                work.replace(destination)
                entries.append(dict(id=demo_id, path=f"demos/{demo_id}"))
                attach(parent, entries)
                record["id"] = demo_id
            else:
                failed = output / "failed" / parent.name / f"t{trial:02d}"
                failed.parent.mkdir(parents=True, exist_ok=True)
                if failed.exists():
                    raise FileExistsError(failed)
                # Keep diagnostic videos, poses and controls, not unusable dense training arrays.
                for filename in ("observation.npz", "supervision.npz", "tcow_labels.npz"):
                    (work / filename).unlink()
                (work / "README.md").write_text(description.replace("Status: collecting", "Status: failed; excluded from training") +
                                              f"Failure: {record['failure_reason']}\nDense training arrays removed; diagnostic streams retained.\n")
                work.replace(failed)
            state["trials"].append(record)
            state["demonstrations"] = entries
            write_json(state_path, state)
            print(json.dumps(dict(event="trial_finished", path=row["path"], **record)),
                  file=sys.__stdout__, flush=True)
        state["status"] = "complete" if MIN_DEMOS <= len(entries) <= maximum else "shortfall"
        state["added"] = len(entries)
        state["failed_trials"] = sum(not trial["success"] for trial in state["trials"])
        write_json(state_path, state)
    return state


def augment_rows(root, rows, output, workers=4, extra_demos=MAX_DEMOS, resume=False):
    """Attach demos to standard/train rows of either a new or existing dataset."""
    root, output = root.resolve(), output.resolve()
    if not 1 <= workers <= 4:
        raise ValueError("use 1–4 workers within the eight-CPU budget")
    if extra_demos not in (2, 3):
        raise ValueError("request two or three extra demonstrations")
    representation = json.loads((root / "normalization.json").read_text())["action_representation"]
    if representation not in ("relative_joint", "relative_ee"):
        raise ValueError("extra demonstrations require joint-delta or EE-delta labels")
    rows = [row for row in rows if row["group"] == "standard" and row["split"] == "train"]
    if not rows:
        raise ValueError("no standard/train samples to augment")
    for row in rows:
        meta = json.loads((root / row["path"] / "episode.json").read_text())
        if (not meta["success"] or (meta["seed"], meta["layout"], meta["split"]) !=
                (row["seed"], "standard", "train") or
                meta["context_plan"] != make_context_plan(meta["seed"], meta["swaps"])):
            raise ValueError(f"parent does not match current scene generator: {row['path']}")
    config = dict(data=str(root), action_representation=representation,
                  train_samples=len(rows), minimum_extra=MIN_DEMOS,
                  maximum_extra=extra_demos, maximum_trials=MAX_TRIALS, settings=SETTINGS,
                  profiles=PROFILES, schedule="mild/medium/strong, then alternating mild/medium",
                  horizon=25, stride=10, fps=25, workers=workers,
                  manifest_sha256=digest(root / "manifest.jsonl"),
                  normalization_sha256=digest(root / "normalization.json"), training="not run")
    if resume:
        if json.loads((output / "config.json").read_text()) != config:
            raise ValueError("resume configuration differs from original run")
    else:
        if any((root / row["path"] / "demonstrations.json").exists() for row in rows):
            raise ValueError("training samples already have attached demonstrations")
        output.mkdir(parents=True, exist_ok=False)
        for name in ("scenes", "scene_logs", "failed"):
            (output / name).mkdir()
        write_json(output / "config.json", config)
        (output / "README.md").write_text(
            f"# {output.name}\n\nAttach {MIN_DEMOS}–{extra_demos} successful physical oracle + perturbation demonstrations to each of {len(rows)} train samples.\n"
            "Continuous 25 Hz correlated arm-joint noise from shuffle end; executed-action labels.\n"
            "Gripper unchanged; max3 physical recovery attempts at noise scale0.2; no pose resets.\n"
            "Up to12 rollout trials/sample: mild, medium, strong, then alternate mild/medium.\n"
            f"Each demo has its own RGB/wrist/proprio (no depth) history; {representation} H25, stride10.\n"
            "Train data: standard/train parent episodes. Original scene splits and normalization retained.\n"
            "Training/weights/epochs/batch/LR/frozen modules: not applicable; no FM training or inference.\n"
            "Test: per-demo full episode alignment, scene identity and exact executed-control labels checked before attachment.\n"
            f"{workers} workers; run within one EGL GPU/eight CPU cores. Failed demos excluded, diagnostic videos retained.\n"
            "Config: config.json. Progress: summary.json, scenes/*.json, scene_logs/*.log. Status: collecting.\n")
    results = []
    started = time.monotonic()
    print(json.dumps(dict(event="augmentation_start", **config)), flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool, \
            tqdm(total=len(rows), desc="augment train samples", unit="sample", mininterval=5, file=sys.stdout) as progress:
        # Bound queued work: unexpected errors stop collection after the active batch.
        for offset in range(0, len(rows), workers):
            futures = [pool.submit(collect_scene, (root, row, output), extra_demos)
                       for row in rows[offset:offset + workers]]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                summary = dict(status="running", samples=len(rows), completed=len(results),
                               added=sum(r["added"] for r in results),
                               shortfalls=[r["path"] for r in results if r["status"] == "shortfall"],
                               elapsed_s=time.monotonic() - started)
                write_json(output / "summary.json", summary)
                progress.update(1)
                print(json.dumps(dict(event="sample_complete", path=result["path"], added=result["added"],
                                      failed_trials=result["failed_trials"], completed=len(results), total=len(rows))), flush=True)
    if digest(root / "manifest.jsonl") != config["manifest_sha256"] or digest(root / "normalization.json") != config["normalization_sha256"]:
        raise ValueError("original split manifest or normalization changed during collection")
    expanded = expand_demonstrations(root, rows)
    summary.update(status="shortfall" if summary["shortfalls"] else "complete",
                   demonstrations_per_epoch=len(expanded), base_demonstrations=len(rows),
                   failed_trials=sum(r["failed_trials"] for r in results))
    write_json(output / "summary.json", summary)
    readme = output / "README.md"
    readme.write_text(readme.read_text() + "\nFinal result:\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")
    print(json.dumps(dict(event="augmentation_finished", **summary)), flush=True)
    if summary["shortfalls"]:
        raise RuntimeError(f"{len(summary['shortfalls'])} samples have fewer than two added demonstrations; see summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--extra-demos", type=int, choices=(2, 3), default=MAX_DEMOS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows, _ = split_rows(args.data)
    augment_rows(args.data, rows, args.output, args.workers, args.extra_demos, args.resume)


if __name__ == "__main__":
    main()
