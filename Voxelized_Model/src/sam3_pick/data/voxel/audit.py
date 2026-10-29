"""Audit saved 50 Hz voxel trajectories and replay a deterministic subset."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
import json
from pathlib import Path
import mujoco
import numpy as np
from tqdm import tqdm
from sam3_pick.simulation.robot_scene import (
    assert_robot_table_only,
    assert_user_arm_configuration,
)

AUDIT = {}


def initialize_audit(dataset):
    AUDIT["dataset"] = Path(dataset)
    AUDIT["config"] = json.loads((AUDIT["dataset"] / "config.json").read_text())
    AUDIT["model"] = mujoco.MjModel.from_xml_path(str(AUDIT["dataset"] / "scene.xml"))
    assert_robot_table_only(AUDIT["model"])
    assert_user_arm_configuration(AUDIT["model"])
    AUDIT["site"] = AUDIT["model"].site("grasp_site").id


def audit_episode(job):
    record, expected_target, should_replay = job
    dataset, config, model, site = (AUDIT[name] for name in ("dataset", "config", "model", "site"))
    replay_error = None
    with np.load(dataset / record["trajectory_file"]) as episode:
        actions = episode["action_absolute"]
        states = episode["state_full_qpos"]
        velocities = episode["state_full_qvel"]
        ee = episode["ee_position_world_m"]
        phase = episode["phase"]
        target = episode["target_world_m"]
        frames = len(actions)
        assert actions.shape == (frames, 6)
        assert states.shape == (frames + 1, model.nq)
        assert velocities.shape == (frames + 1, model.nv)
        assert episode["state_joint_position"].shape == (frames + 1, 6)
        assert ee.shape == (frames + 1, 3)
        assert phase.shape == (frames,)
        assert np.isfinite(actions).all() and np.isfinite(states).all()
        assert np.allclose(np.diff(episode["timestamp_action_s"]), 0.02, rtol=0, atol=1e-12)
        assert np.allclose(np.diff(episode["timestamp_state_s"]), 0.02, rtol=0, atol=1e-12)
        assert episode["timestamp_state_s"][-1] == frames / 50
        assert np.array_equal(target, expected_target)
        assert np.all(actions >= model.actuator_ctrlrange[:, 0] - 1e-06)
        assert np.all(actions <= model.actuator_ctrlrange[:, 1] + 1e-06)
        assert np.all(np.diff(phase.astype(int)) >= 0)
        assert set(phase) >= {0, 1, 2, 3}
        start_close = int(np.flatnonzero(phase == 2)[0])
        assert np.all(actions[phase < 2, 5] == 0)
        assert np.all(np.diff(actions[phase >= 2, 5]) <= 1e-07)
        assert np.allclose(actions[phase == 3, 5], -0.0255, atol=1e-07)
        point_errors = np.linalg.norm(ee - target, axis=1)
        assert point_errors[start_close] <= config["arrival_final_tolerance_m"] + 1e-06
        assert point_errors[-1] <= config["arrival_final_tolerance_m"] + 1e-06
        assert point_errors[start_close:].max() <= config["closing_hold_tolerance_m"] + 1e-06
        assert abs(episode["state_joint_position"][-1, -1] + 0.0255) <= 0.0005
        final_error = float(point_errors[-1])
        phases = Counter(phase.tolist())
        if should_replay:
            data = mujoco.MjData(model)
            data.qpos[:] = states[0]
            data.qvel[:] = velocities[0]
            data.ctrl[:] = episode["initial_ctrl"]
            mujoco.mj_forward(model, data)
            maximum = 0.0
            for j, action in enumerate(actions):
                data.ctrl[:] = action
                for _ in range(10):
                    mujoco.mj_step(model, data)
                    assert all(
                        (
                            c.dist >= -config["physics_penetration_tolerance_m"] - 1e-06
                            for c in data.contact
                        )
                    )
                mujoco.mj_forward(model, data)
                maximum = max(maximum, float(np.linalg.norm(data.site_xpos[site] - ee[j + 1])))
            assert maximum <= 1e-05, f"Replay mismatch {maximum} m"
            replay_error = maximum
    return (final_error, replay_error, phases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--replay", type=int, default=16)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("Require 1..32 audit workers")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[: args.workers])
    config = json.loads((args.dataset / "config.json").read_text())
    summary = json.loads((args.dataset / "summary.json").read_text())
    assert config["status"] == summary["status"] == "complete"
    assert config["fps"] == 50
    with np.load(args.dataset / "grid.npz") as grid:
        targets = grid["centers_world_m"].copy()
        status = grid["status"].copy()
        assert int(np.count_nonzero(status == 2)) == summary.get("accepted", 0)
        assert int(np.count_nonzero(status >= 0)) == summary["tested"]
    records = [json.loads(line) for line in (args.dataset / "index.jsonl").read_text().splitlines()]
    assert len(records) == summary["tested"]
    assert len({row["voxel_id"] for row in records}) == len(records)
    accepted = [row for row in records if row["trajectory_success"]]
    assert len(accepted) == summary.get("accepted", 0)
    expected_files = {row["trajectory_file"] for row in accepted}
    actual_files = {
        str(path.relative_to(args.dataset))
        for path in (args.dataset / "trajectories").glob("*.npz")
    }
    assert actual_files == expected_files, (
        f"Trajectory/index mismatch: {actual_files ^ expected_files}"
    )
    replay_indices = set(
        np.linspace(0, len(accepted) - 1, min(args.replay, len(accepted)), dtype=int)
    )
    errors, replay_errors, phase_counts = ([], [], Counter())
    jobs = ((row, targets[row["voxel_id"]], i in replay_indices) for i, row in enumerate(accepted))
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize_audit,
        initargs=(str(args.dataset),),
    ) as pool:
        for final_error, replay_error, phases in tqdm(
            pool.map(audit_episode, jobs, chunksize=16),
            total=len(accepted),
            desc="Audit 50 Hz data",
            mininterval=2,
        ):
            errors.append(final_error)
            phase_counts.update(phases)
            if replay_error is not None:
                replay_errors.append(replay_error)
    assert sum(phase_counts.values()) == summary.get("frames", 0)
    report = {
        "status": "passed",
        "audited_episodes": len(accepted),
        "audited_frames": sum(phase_counts.values()),
        "replayed_episodes": len(replay_errors),
        "max_replay_ee_difference_m": max(replay_errors) if replay_errors else None,
        "max_final_ee_error_m": max(errors) if errors else None,
        "mean_final_ee_error_m": float(np.mean(errors)) if errors else None,
        "phase_frame_counts": dict(phase_counts),
        "checks": "grid targets, 50 Hz timestamps, absolute command bounds, pre-close arrival, closure/hold EE drift, closed gripper, physical replay and substep collision checks",
    }
    (args.dataset / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
