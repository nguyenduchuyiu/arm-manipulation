"""Export audited voxel trajectories as self-contained H25 EE-delta chunks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import mujoco
import numpy as np
from tqdm import tqdm
from sam3_pick.simulation.pose import (
    ACTION_DIM,
    REPRESENTATION,
    STATE_DIM,
    STATE_REPRESENTATION,
    UNITS,
    GRIPPER_NORMALIZATION,
    base_state,
    measured_rotations,
    pose_labels,
)

SOURCE = {}


def initialize(dataset, origin):
    SOURCE.update(
        root=Path(dataset),
        origin=np.asarray(origin),
        model=mujoco.MjModel.from_xml_path(str(Path(dataset) / "scene.xml")),
    )


def episode_chunks(row):
    with np.load(SOURCE["root"] / row["trajectory_file"]) as z:
        commands = z["action_absolute"]
        ee, qpos = (z["ee_position_world_m"], z["state_full_qpos"])
        joints = z["state_joint_position"]
        total = len(commands)
        assert total == row["frames"] and ee.shape == (total + 1, 3)
        np.testing.assert_array_equal(z["target_world_m"], row["target_world_m"])
        np.testing.assert_array_equal(z["timestamp_action_s"], np.arange(total) / 50)
        np.testing.assert_array_equal(z["timestamp_state_s"], np.arange(total + 1) / 50)
        starts = np.arange(0, (total + 1) // 2, 2, dtype=np.int32)
        rotations = measured_rotations(SOURCE["model"], qpos, ee)
        actions, mask, anchor, anchor_r, _ = pose_labels(ee, rotations, commands, starts)
        jaw = joints[2 * starts, 5]
        jaw_address = SOURCE["model"].jnt_qposadr[SOURCE["model"].joint("right_jaw_slide_joint").id]
        np.testing.assert_allclose(jaw, qpos[2 * starts, jaw_address], atol=2e-09, rtol=0)
        state = base_state(anchor, anchor_r, SOURCE["origin"], jaw).astype(np.float32)
        assert state.shape == (len(starts), STATE_DIM) and np.isfinite(state).all()
        np.testing.assert_array_equal(state[:, 9], jaw)
        indices = np.minimum(2 * (starts[:, None] + np.arange(25)), 2 * ((total - 1) // 2))
        np.testing.assert_array_equal(actions[..., 9], commands[indices, 5])
        np.testing.assert_array_equal(mask, 2 * (starts[:, None] + np.arange(25)) < total)
    return (actions, mask, starts, anchor, anchor_r, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError("Require 1..32 workers")
    source_config = json.loads((args.dataset / "config.json").read_text())
    source_audit = json.loads((args.dataset / "audit.json").read_text())
    assert source_config["status"] == "complete" and source_config["fps"] == 50
    assert source_audit["status"] == "passed"
    rows = [json.loads(line) for line in (args.dataset / "index.jsonl").read_text().splitlines()]
    rows = sorted((r for r in rows if r["trajectory_success"]), key=lambda r: r["voxel_id"])
    assert rows and len({r["voxel_id"] for r in rows}) == len(rows)
    assert len(rows) == source_audit["audited_episodes"]
    lengths = np.array([(r["frames"] + 1) // 2 for r in rows], np.int32)
    offsets = np.r_[0, np.cumsum((lengths.astype(np.int64) + 1) // 2)]
    count = int(offsets[-1])
    origin = np.array(source_config["base_origin_world_m"])
    args.output.mkdir(parents=True, exist_ok=False)
    readme = f"# H25 EE-delta chunks\n\nStatus: exporting.\n\nSource: `{args.dataset.resolve()}`; physically verified voxel trajectories only. 25Hz targets, H25, chunk-start stride2. Actions[N,25,10]: measured future EE world/base-axis delta XYZ3, relative rotation6D6, absolute jaw command1. Pose anchor is the measured EE at source state2*start, fixed for the entire chunk. Target k is source state min(2*(start+k+1),T). Relative rotation = R_future @ R_anchor.T; first two columns stored column-major. Decode with Gram-Schmidt, then R_target=delta_R @ R_anchor. No accumulated deltas.\n\nObservation state_ee_pose_base[N,10]: base XYZ3, absolute rotation6D6, measured right-jaw position1 at chunk start. Base axes align with world; position_base=position_world-base_origin. Goal XYZ saved in both frames. No arm joint input. Gripper observation is measured position; gripper action is the original absolute actuator command. All physical values are stored in metres, rotation6D unitless. Jaw0=open,-0.0255=closed.\n\nMask padding in training loss; padding repeats the final label. Chunks never cross trajectories. Source trajectories were physically checked at50Hz; export does not claim a new25Hz dynamic verification.\n\nNo training, source weights, optimizer, frozen modules, epochs, batch or LR is applied here. No learned checkpoint, K, flow steps, noise seed or policy frame budget. Training should compute pose/goal normalization on its training split only. Gripper uses fixed range g_norm=-1-2*g/0.0255 inside the model, inverse g=-0.0255*(g_norm+1)/2; it remains absolute.\n\nFiles: actions.npy,action_mask.npy,state_ee_pose_base.npy,anchor_ee.npy,anchor_rotation.npy,chunk_start_25hz.npy,chunk_trajectory_index.npy,target_world_m.npy,target_base_m.npy,trajectory_voxel_id.npy,trajectory_length_25hz.npy,trajectory_chunk_offsets.npy,trajectories.jsonl,scene.xml,config.json,audit.json. NumPy arrays support memory mapping.\n\nExact configuration/provenance: [config.json](config.json). Export audit: [audit.json](audit.json).\n"
    (args.output / "README.md").write_text(readme)
    config = dict(
        status="exporting",
        source_dataset=str(args.dataset.resolve()),
        episodes=len(rows),
        chunks=count,
        source_hz=50,
        sample_hz=25,
        horizon=25,
        chunk_stride=2,
        action_dim=ACTION_DIM,
        action_representation=REPRESENTATION,
        action_units=UNITS,
        state_dim=STATE_DIM,
        state_representation=STATE_REPRESENTATION,
        base_origin_world_m=origin.tolist(),
        base_axes="aligned with world",
        gripper_storage="absolute metres, no delta; observation measured, action commanded",
        training_gripper_normalization=GRIPPER_NORMALIZATION,
        normalization_applied=False,
        rotation="R_future @ R_anchor.T, first two columns, column-major",
        target_state="min(2*(start+k+1),T); anchor state2*start; k=0..24",
        source_hashes={
            name: hashlib.sha256((args.dataset / name).read_bytes()).hexdigest()
            for name in ("config.json", "index.jsonl", "scene.xml", "audit.json")
        },
        workers=args.workers,
        training="not run",
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    shapes = dict(
        actions=((count, 25, ACTION_DIM), np.float32),
        action_mask=((count, 25), np.bool_),
        state_ee_pose_base=((count, STATE_DIM), np.float32),
        anchor_ee=((count, 3), np.float32),
        anchor_rotation=((count, 3, 3), np.float64),
        chunk_start_25hz=((count,), np.int32),
        chunk_trajectory_index=((count,), np.int32),
    )
    arrays = {
        name: np.lib.format.open_memmap(
            args.output / f"{name}.npy", mode="w+", shape=shape, dtype=dtype
        )
        for name, (shape, dtype) in shapes.items()
    }
    full, valid_slots = (0, 0)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=initialize,
        initargs=(args.dataset, origin),
    ) as pool:
        results = pool.map(episode_chunks, rows, chunksize=8)
        with (args.output / "trajectories.jsonl").open("w") as manifest:
            for ep, (row, values) in enumerate(
                tqdm(
                    zip(rows, results),
                    total=len(rows),
                    desc="Export/audit EE-delta chunks",
                    mininterval=2,
                    file=sys.stdout,
                )
            ):
                action, mask, starts, anchor, rotation, state = values
                lo, hi = (int(offsets[ep]), int(offsets[ep + 1]))
                assert hi - lo == len(starts)
                for name, value in dict(
                    actions=action,
                    action_mask=mask,
                    chunk_start_25hz=starts,
                    anchor_ee=anchor,
                    anchor_rotation=rotation,
                    state_ee_pose_base=state,
                ).items():
                    arrays[name][lo:hi] = value
                arrays["chunk_trajectory_index"][lo:hi] = ep
                full += int(mask.all(1).sum())
                valid_slots += int(mask.sum())
                entry = dict(
                    trajectory_index=ep,
                    voxel_id=row["voxel_id"],
                    source_trajectory_file=row["trajectory_file"],
                    source_frames_50hz=row["frames"],
                    frames_25hz=int(lengths[ep]),
                    chunk_begin=lo,
                    chunk_end=hi,
                    target_world_m=row["target_world_m"],
                )
                manifest.write(json.dumps(entry) + "\n")
    for array in arrays.values():
        array.flush()
    goals = np.array([row["target_world_m"] for row in rows])
    for name, values in dict(
        target_world_m=goals,
        target_base_m=goals - origin,
        trajectory_voxel_id=np.array([row["voxel_id"] for row in rows], np.int32),
        trajectory_length_25hz=lengths,
        trajectory_chunk_offsets=offsets,
    ).items():
        np.save(args.output / f"{name}.npy", values)
    shutil.copyfile(args.dataset / "scene.xml", args.output / "scene.xml")
    expected_full = sum((max((int(n) - 25) // 2 + 1, 0) for n in lengths))
    expected_valid = sum(
        (sum((min(25, int(n) - start) for start in range(0, int(n), 2))) for n in lengths)
    )
    assert full == expected_full and valid_slots == expected_valid
    for name, (shape, dtype) in shapes.items():
        saved = np.load(args.output / f"{name}.npy", mmap_mode="r")
        assert saved.shape == shape and saved.dtype == dtype
    audit = dict(
        status="passed",
        audited_episodes=len(rows),
        audited_chunks=count,
        full_chunks=full,
        padded_chunks=count - full,
        valid_action_slots=valid_slots,
        masked_action_slots=count * 25 - valid_slots,
        source_frames_50hz=sum((r["frames"] for r in rows)),
        checks="all pose labels reconstructed against measured positions/orientations; measured-qpos FK; anchor jaw vs measured state; original absolute gripper commands; timestamps; mask bounds; tail padding; episode offsets; shapes; independent count formulas",
    )
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    config["status"] = "complete"
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "README.md").write_text(
        readme.replace("Status: exporting.", "Status: complete; audit passed.")
        + f"\nCompleted:{count:,}chunks from{len(rows):,}trajectories;{full:,}full,{count - full:,}padded. All label/mask/state audits passed.\n"
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
