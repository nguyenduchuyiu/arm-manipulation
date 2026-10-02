"""Render synchronized wrist videos from saved poses, retaining original data read-only."""
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
import sys
import time

import av
import imageio.v3 as iio
import mujoco
import numpy as np
from tqdm.auto import tqdm

from controllers.nexarm_mujoco_backend import MUJOCO_JOINTS
from memory_occlusion.environment.env import SCENE_XML, configure_mvp_model, wrist_camera_metadata


def physical_positions(state, limits):
    fraction = state.astype(np.float64).copy()
    fraction[..., :5] = (fraction[..., :5] + 1) / 2
    return limits[:, 0] + fraction * np.diff(limits, axis=1)[:, 0]


def restore_pose(model, data, joints, joint_addresses, poses, pose_addresses, timestamp, qpos=None):
    if qpos is not None:
        data.qpos[:] = qpos
    else:
        data.qpos[joint_addresses] = joints
        for address, pose in zip(pose_addresses, poses.reshape(-1, 7)):
            data.qpos[address:address + 7] = pose
        right = model.jnt_qposadr[model.joint("right_jaw_slide_joint").id]
        left = model.jnt_qposadr[model.joint("left_jaw_slide_joint").id]
        pinion = model.jnt_qposadr[model.joint("gripper_pinion_joint").id]
        # Old data recorded the actual right jaw, but not the two passive joints.
        opposed = model.eq_data[model.equality("gripper_jaws_opposed_1_to_1").id, :5]
        geared = model.eq_data[model.equality("gripper_pinion_to_right_jaw").id, :5]
        if not np.array_equal(opposed, [0, -1, 0, 0, 0]):
            raise ValueError("unsupported opposed-jaw constraint")
        data.qpos[left] = model.qpos0[left] - (data.qpos[right] - model.qpos0[right])
        data.qpos[pinion] = model.qpos0[pinion] + np.polynomial.polynomial.polyval(
            data.qpos[right] - model.qpos0[right], geared)
    data.time = float(timestamp)
    # Forward kinematics and camera/light poses only; no dynamics or IK replay.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)


def boundary_error(predicted, truth):
    def expand(mask):
        padded = np.pad(mask, 2)
        height, width = mask.shape
        return np.logical_or.reduce([padded[y:y + height, x:x + width]
                                     for y in range(5) for x in range(5)])
    bad = (predicted & ~expand(truth)).sum() + (truth & ~expand(predicted)).sum()
    return float(bad / max(int(truth.sum() + predicted.sum()), 1))


def gate_frames(meta):
    selected = {0, meta["frames"] - 1, meta["decision_frames"]["t_occ"] - 1}
    selected.update(meta["decision_frames"].values())
    selected.update(meta["grasp_frames"].values())
    selected.update(meta["semantic_transition_frames"].values())
    for interval in meta["phase_intervals"]:
        if interval["phase"] in ("cover_close", "object_close", "cover_approach", "object_approach"):
            selected.update((interval["start_frame"], interval["end_frame_exclusive"] - 1))
    return sorted(selected)


def start_worker():
    global MODEL, DATA, RGB_RENDERER, SEG_RENDERER
    MODEL = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    configure_mvp_model(MODEL)
    DATA = mujoco.MjData(MODEL)
    RGB_RENDERER = mujoco.Renderer(MODEL, height=320, width=320)
    samples = MODEL.vis.quality.offsamples
    MODEL.vis.quality.offsamples = 0
    SEG_RENDERER = mujoco.Renderer(MODEL, height=320, width=320)
    MODEL.vis.quality.offsamples = samples
    SEG_RENDERER.enable_segmentation_rendering()
    atexit.register(RGB_RENDERER.close)
    atexit.register(SEG_RENDERER.close)


def render_episode(source, output, row):
    started = time.perf_counter()
    path, destination = source / row["path"], output / row["path"]
    meta = json.loads((path / "episode.json").read_text())
    n = meta["frames"]
    if meta["fps"] != 25 or meta["resolution"] != [240, 320]:
        raise ValueError(f"expected 25 Hz 240x320 episode: {path}")
    with np.load(path / "observation.npz") as archive:
        state, timestamps = archive["joint_position"], archive["timestamp_s"]
    with np.load(path / "debug_poses.npz") as archive:
        poses, names = archive["poses"], archive["body_names"].tolist()
        if names != meta["pose_body_order"] or not np.array_equal(archive["timestamp_s"], timestamps):
            raise ValueError(f"pose timestamps/body order differ: {path}")
        full_qpos = archive["qpos"] if "qpos" in archive else None
    if state.shape != (n, 6) or poses.shape != (n, len(names) * 7):
        raise ValueError(f"pose/state frame counts differ: {path}")
    if full_qpos is not None and full_qpos.shape != (n, MODEL.nq):
        raise ValueError(f"saved full qpos does not match scene model: {path}")
    if not np.allclose(timestamps, np.arange(n) / 25, atol=1e-9):
        raise ValueError(f"non-25 Hz timestamps: {path}")
    limits = np.asarray(meta["normalization_limits"])
    if not np.allclose(limits, MODEL.actuator_ctrlrange, rtol=0, atol=1e-9):
        raise ValueError(f"joint normalization limits differ: {path}")
    joints = physical_positions(state, limits)
    joint_addresses = [MODEL.jnt_qposadr[MODEL.joint(MUJOCO_JOINTS[name]).id] for name in meta["joint_names"]]
    pose_addresses = [MODEL.jnt_qposadr[MODEL.joint(name + "_joint").id] for name in names]
    overview = MODEL.camera("overview")
    overview.pos[:] = meta["overview_camera"]["position"]
    overview.quat[:] = meta["overview_camera"]["quaternion"]
    overview.fovy[:] = meta["overview_camera"]["fovy"]
    selected = gate_frames(meta)
    original = {}
    for index, rgb in enumerate(iio.imiter(path / "rgb.mp4", plugin="pyav")):
        if index in selected:
            original[index] = rgb
    if index + 1 != n or len(original) != len(selected):
        raise ValueError(f"original RGB frame count differs: {path}")
    with np.load(path / "supervision.npz") as archive:
        labels = archive["mask"][selected].copy()
        entity = archive["semantic_entity"][selected].copy()
    gate_labels = dict(zip(selected, zip(labels, entity)))
    target_geom = MODEL.geom(meta["target_object_id"] + "_visual").id
    cover_geoms = [i for i in range(MODEL.ngeom)
                   if MODEL.geom(i).name.startswith(meta["correct_cover_body"] + "_")]
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("rgb.mp4", "observation.npz", "supervision.npz", "tcow_labels.npz",
                 "query_mask.png", "debug_poses.npz"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
        if not (destination / name).is_symlink():
            (destination / name).symlink_to((path / name).resolve())
    if (path / "relative_actions.npz").is_file() and not (destination / "relative_actions.npz").is_symlink():
        (destination / "relative_actions.npz").symlink_to((path / "relative_actions.npz").resolve())
    original_directory = (path / "rgb.mp4").resolve().parent
    for snapshot in original_directory.glob("*.png"):
        if not (destination / snapshot.name).is_symlink():
            (destination / snapshot.name).symlink_to(snapshot)
    gates = []
    temporary = destination / "wrist_rgb.partial.mp4"
    mujoco.mj_resetData(MODEL, DATA)
    top = int(meta["overview_camera"]["crop_top"])
    with iio.imopen(temporary, "w", plugin="pyav") as writer:
        writer.init_video_stream("libx264", fps=25, pixel_format="yuv420p")
        for frame in tqdm(range(n), desc=f"wrist {path.name}", unit="frame", mininterval=5,
                          leave=False, file=sys.stdout):
            restore_pose(MODEL, DATA, joints[frame], joint_addresses, poses[frame], pose_addresses,
                         timestamps[frame], full_qpos[frame] if full_qpos is not None else None)
            if frame in original:
                RGB_RENDERER.update_scene(DATA, camera="overview")
                rebuilt = RGB_RENDERER.render()[top:top + 240]
                rgb_mae = float(np.abs(rebuilt.astype(np.float32) - original[frame]).mean())
                SEG_RENDERER.update_scene(DATA, camera="overview")
                segmentation = SEG_RENDERER.render()[top:top + 240, :, 0]
                mask, active = gate_labels[frame]
                prediction = (segmentation == target_geom if active == 2 else
                              np.isin(segmentation, cover_geoms) if active == 1 else np.zeros_like(mask, bool))
                error = boundary_error(prediction, mask > 0)
                gates.append({"frame": frame, "rgb_mae_255": rgb_mae,
                              "mask_error_outside_2px": error})
                if rgb_mae > 3 or error > .02:
                    raise ValueError(f"reconstruction gate failed: {path.name}, frame {frame}, "
                                     f"RGB MAE={rgb_mae:.3f}, mask error={error:.4f}")
            RGB_RENDERER.update_scene(DATA, camera="wrist")
            writer.write(RGB_RENDERER.render(), is_batch=False)
    with av.open(str(temporary)) as video:
        stream = video.streams.video[0]
        if stream.frames != n or stream.average_rate != 25 or (stream.height, stream.width) != (320, 320):
            raise ValueError(f"wrist video is not aligned: {destination}")
    temporary.replace(destination / "wrist_rgb.mp4")
    method = "full_qpos" if full_qpos is not None else "recorded_proprio_and_object_poses_with_geometric_passive_joints"
    meta["wrist_camera"] = {**wrist_camera_metadata(MODEL, n), "reconstruction": method}
    (destination / "episode.json").write_text(json.dumps(meta, indent=2) + "\n")
    (destination / "input.json").write_text(json.dumps({"rgb_video": "rgb.mp4", "wrist_rgb_video": "wrist_rgb.mp4",
        "observation": "observation.npz", "query_rgb": meta["query_rgb"], "decision_frames": meta["decision_frames"]}, indent=2) + "\n")
    result = {"path": row["path"], "frames": n, "seconds": time.perf_counter() - started,
              "reconstruction": method, "gates": gates,
              "max_rgb_mae_255": max(g["rgb_mae_255"] for g in gates),
              "max_mask_error_outside_2px": max(g["mask_error_outside_2px"] for g in gates)}
    (destination / "wrist_backfill.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=0, help="small real-data smoke; 0 processes the full manifest")
    parser.add_argument("--resume", action="store_true", help="resume only this source and output configuration")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.max_episodes < 0:
        raise ValueError("use 1–8 workers and nonnegative max-episodes")
    source = args.source.resolve()
    manifest = (source / "manifest.jsonl").read_bytes()
    rows = [json.loads(line) for line in manifest.decode().splitlines()]
    rows = rows[:args.max_episodes] if args.max_episodes else rows
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("empty or duplicated episode manifest")
    config = {"source": str(source), "source_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
              "episodes": len(rows), "max_episodes": args.max_episodes,
              "fps": 25, "wrist_resolution": [320, 320],
              "method": "saved-state wrist rendering; no dynamics/IK; linked read-only original data"}
    if args.resume:
        if json.loads((args.output / "backfill_config.json").read_text()) != config:
            raise ValueError("resume source/manifest/config differs")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "backfill_config.json").write_text(json.dumps(config, indent=2) + "\n")
    for group in sorted({str(Path(row["path"]).parent) for row in rows}):
        first = next(row for row in rows if str(Path(row["path"]).parent) == group)
        references = (source / first["path"] / "rgb.mp4").resolve().parent.parent / "references"
        if not references.is_dir():
            raise FileNotFoundError(references)
        destination = args.output / group
        destination.mkdir(parents=True, exist_ok=True)
        if not (destination / "references").is_symlink():
            (destination / "references").symlink_to(references)
    results, pending = [], []
    for row in rows:
        completed = args.output / row["path"] / "wrist_backfill.json"
        if completed.is_file() and (completed.parent / "wrist_rgb.mp4").is_file():
            results.append(json.loads(completed.read_text()))
        else:
            pending.append(row)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), initializer=start_worker) as pool:
        futures = [pool.submit(render_episode, source, args.output, row) for row in pending]
        for future in tqdm(as_completed(futures), total=len(futures), desc="backfill episodes", unit="episode",
                           mininterval=5, file=sys.stdout):
            result = future.result()
            results.append(result)
            tqdm.write(json.dumps({"event": "wrist_episode_ready", "path": result["path"],
                                   "frames": result["frames"], "seconds": round(result["seconds"], 2),
                                   "max_rgb_mae_255": result["max_rgb_mae_255"],
                                   "max_mask_error_outside_2px": result["max_mask_error_outside_2px"]}), file=sys.stdout)
    for name in ("normalization.json", "plan.jsonl", "plan_summary.json", "standard_seeds.txt", "composition_seeds.txt"):
        if (source / name).is_file():
            shutil.copyfile(source / name, args.output / name)
    source_metadata = json.loads((source / "dataset.json").read_text()) if (source / "dataset.json").is_file() else {}
    (args.output / "dataset.json").write_text(json.dumps({**source_metadata, **config,
        "wrist_camera": True, "action_labels": "unchanged", "source": str(source)}, indent=2) + "\n")
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = {"episodes": len(results), "frames": sum(r["frames"] for r in results),
               "elapsed_seconds": time.perf_counter() - started,
               "max_rgb_mae_255": max(r["max_rgb_mae_255"] for r in results),
               "max_mask_error_outside_2px": max(r["max_mask_error_outside_2px"] for r in results),
               "output": str(args.output)}
    (args.output / "backfill_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "wrist_backfill_complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
