"""Render complete wide-pick episodes with front RGB-D and fisheye wrist RGB."""

from sam3_pick.paths import PROJECT
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import time
import av
import mujoco
import numpy as np
from tqdm import tqdm
import sam3_pick.data.rendering as r
from sam3_pick.simulation.scene import BASE, write_json
from sam3_pick.simulation.wrist_camera import INTRINSICS, REFERENCE
from sam3_pick.data.pick_collect import DATA
from sam3_pick.utils.hashing import sha256_file as digest

SCHEMA = "pick_wide_front_rgbd_wrist_fisheye_rgb_v1"


def initialize(root):
    r.initialize(root)
    atexit.register(r.RGB.close)
    atexit.register(r.SEG.close)
    atexit.register(r.WRIST.close)


def close(writer):
    try:
        writer.stdin.close()
    finally:
        writer.wait()
    assert writer.returncode == 0


def rgb_encoder(path):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "320x240",
            "-r",
            "25",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def render(row):
    directory = r.ROOT / row["path"]
    marker = directory / "render_complete.json"
    if marker.exists():
        result = json.loads(marker.read_text())
        assert result["camera_schema"] == SCHEMA
        return result
    assert shutil.disk_usage(r.ROOT).free >= 3 * 2**30, (
        "Less than3GiB free; free space before resuming."
    )
    with np.load(directory / "sim_state.npz") as z:
        states = {k: z[k] for k in z.files}
    n = len(states["qpos"])
    t = row["target_id"]
    geoms = r.GEOMS[t + 1]
    instance = np.zeros((n, 2, 240, 320), "uint8")
    amodal = np.zeros((n, 2, 240, 40), "uint8")
    fov = np.zeros((n, 2), bool)
    transforms = np.tile(np.eye(4), (n, 2, 1, 1))
    raw_hash = hashlib.sha256()
    absolute_error = 0.0
    count = 0
    max_error = 0.0
    start = time.perf_counter()
    with ExitStack() as stack:
        writers = {v: rgb_encoder(directory / f"rgb_{v}.partial.mp4") for v in r.VIEWS}
        depth_writer = r.depth_encoder(directory / "depth_front.partial.mkv")
        for writer in [*writers.values(), depth_writer]:
            stack.callback(close, writer)
        for frame in tqdm(range(n), desc=f"Pick RGB-D {row['path']}", leave=False, mininterval=15):
            for key in ("qpos", "qvel", "ctrl"):
                getattr(r.STATE, key)[:] = states[key][frame]
            mujoco.mj_forward(r.MODEL, r.STATE)
            for vi, (view, camera) in enumerate(zip(r.VIEWS, r.CAMERAS)):
                cid = r.MODEL.camera(camera).id
                transforms[frame, vi, :3, :3] = r.STATE.cam_xmat[cid].reshape(3, 3) @ np.diag(
                    [1.0, -1.0, -1.0]
                )
                transforms[frame, vi, :3, 3] = r.STATE.cam_xpos[cid]
                if view == "front":
                    r.RGB.update_scene(r.STATE, camera=camera, scene_option=r.OPTION)
                    rgb = r.RGB.render()
                    r.RGB.enable_depth_rendering()
                    depth = r.RGB.render()
                    r.RGB.disable_depth_rendering()
                    valid = np.isfinite(depth) & (depth > 0) & (depth <= r.MAX_DEPTH)
                    codes = np.zeros(depth.shape, "<u2")
                    codes[valid] = np.rint(depth[valid] / r.UNIT).astype("uint16")
                    error = np.abs(codes[valid].astype("float64") * r.UNIT - depth[valid])
                    absolute_error += float(error.sum())
                    count += int(valid.sum())
                    max_error = max(max_error, float(error.max()))
                    assert max_error <= 0.0011
                    payload = codes.tobytes()
                    raw_hash.update(payload)
                    depth_writer.stdin.write(payload)
                    r.SEG.update_scene(r.STATE, camera=camera, scene_option=r.OPTION)
                    seg = r.SEG.render()
                else:
                    rgb = r.WRIST.render(r.STATE, r.OPTION)
                    seg = r.WRIST.render(r.STATE, r.OPTION, segmentation=True)
                writers[view].stdin.write(rgb.tobytes())
                valid_geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
                instance[frame, vi][valid_geom] = r.LUT[seg[..., 0][valid_geom]]
                old = r.MODEL.geom_group[geoms].copy()
                r.MODEL.geom_group[geoms] = 5
                if view == "front":
                    r.SEG.update_scene(r.STATE, camera=camera, scene_option=r.ONLY)
                    target = r.SEG.render()[..., 0] >= 0
                else:
                    target = r.WRIST.render(r.STATE, r.ONLY, segmentation=True)[..., 0] >= 0
                r.MODEL.geom_group[geoms] = old
                assert not np.any((instance[frame, vi] == t + 1) & ~target)
                amodal[frame, vi] = np.packbits(target, axis=-1, bitorder="little")
                fov[frame, vi] = target.any()
    for view in r.VIEWS:
        with av.open(str(directory / f"rgb_{view}.partial.mp4")) as video:
            video.streams.video[0].thread_count = 1
            assert sum((1 for _ in video.decode(video=0))) == n
        (directory / f"rgb_{view}.partial.mp4").replace(directory / f"rgb_{view}.mp4")
    decoded_hash = hashlib.sha256()
    decoded = 0
    with av.open(str(directory / "depth_front.partial.mkv")) as video:
        video.streams.video[0].thread_count = 1
        for image in video.decode(video=0):
            decoded_hash.update(
                image.to_ndarray(format="gray16le").astype("<u2", copy=False).tobytes()
            )
            decoded += 1
    assert decoded == n and decoded_hash.hexdigest() == raw_hash.hexdigest()
    (directory / "depth_front.partial.mkv").replace(directory / "depth_front.mkv")
    assert all((np.count_nonzero(instance[0, 0] == i + 1) >= 5 for i in range(4)))
    prompt = instance[0, 0] == t + 1
    assert prompt.sum() > 20
    np.savez_compressed(
        directory / "prompt.npz", query_mask=prompt, view=np.array("front"), frame=0
    )
    np.savez_compressed(
        directory / "masks.npz",
        instance=np.asfortranarray(instance),
        target_amodal_packed=np.asfortranarray(amodal),
        target_visible_packed=np.packbits(instance == t + 1, axis=-1, bitorder="little"),
        target_in_fov=fov,
        target_id=t,
        views=np.array(r.VIEWS),
        timestamp_s=states["timestamp_s"],
        resolution=np.array([320, 240]),
        bitorder=np.array("little"),
    )
    cid = r.MODEL.camera("cam_front").id
    focal = 120 / np.tan(np.deg2rad(r.MODEL.cam_fovy[cid]) / 2)
    K = np.array([[[focal, 0, 159.5], [0, focal, 119.5], [0, 0, 1]], INTRINSICS])
    base = transforms.copy()
    base[..., :3, 3] -= BASE
    np.savez_compressed(
        directory / "calibration.npz",
        intrinsics=K,
        distortion_coefficients=np.zeros((2, 4)),
        projection_model=np.array(["pinhole", "fisheye_equidistant"]),
        camera_to_world=transforms,
        camera_to_base=base,
        timestamp_s=states["timestamp_s"],
        views=np.array(r.VIEWS),
        depth_views=np.array(["front"]),
        depth_unit_m=r.UNIT,
        invalid_depth_code=0,
        depth_max_m=r.MAX_DEPTH,
        render_clip_near_m=r.CLIP_NEAR,
        render_clip_far_m=r.CLIP_FAR,
        resolution=np.array([320, 240]),
        visualization_rotation_deg=np.array([0, 180]),
        wrist_reference=np.array(REFERENCE),
        depth_model=np.array("ideal_geometric_optical_z; sensor noise/focus not simulated"),
    )
    write_json(
        directory / "input.json",
        dict(
            views=list(r.VIEWS),
            target_id=t,
            prompt="prompt.npz",
            rgb={v: f"rgb_{v}.mp4" for v in r.VIEWS},
            depth={"front": "depth_front.mkv"},
            calibration="calibration.npz",
            observations="observation.npz",
            prompt_source="front frame0 only; later masks/XYZ are supervision",
        ),
    )
    result = dict(
        path=row["path"],
        frames=n,
        camera_schema=SCHEMA,
        elapsed_s=time.perf_counter() - start,
        rgb_frames_verified=n,
        depth_codec_lossless=True,
        depth_sha256_decoded=decoded_hash.hexdigest(),
        quantization_max_mm=max_error * 1000,
        quantization_mae_mm=absolute_error / count * 1000,
        bytes=sum((p.stat().st_size for p in directory.iterdir() if p.is_file())),
    )
    write_json(marker, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    root = args.data
    config = json.loads((root / "config.json").read_text())
    assert config["status"] in (
        "physics_complete",
        "rendering",
        "render_complete",
        "complete",
    )
    sources = [
        Path(__file__).resolve(),
        PROJECT / "src/sam3_pick/data/rendering.py",
        PROJECT / "src/sam3_pick/simulation/wrist_camera.py",
        PROJECT / "src/sam3_pick/simulation/render.py",
    ]
    hashes = {str(s): digest(s) for s in sources}
    if "render_source_sha256" in config:
        assert config["render_source_sha256"] == hashes
    config.update(status="rendering", render_source_sha256=hashes)
    write_json(root / "config.json", config)
    rows = [json.loads(s) for s in (root / "manifest.jsonl").read_text().splitlines()]
    results = []
    start = time.perf_counter()
    with ProcessPoolExecutor(
        args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=initialize,
        initargs=(str(root),),
    ) as pool:
        futures = [pool.submit(render, row) for row in rows]
        for future in tqdm(
            as_completed(futures),
            total=len(rows),
            desc="Wide pick front RGB-D + wrist RGB",
            mininterval=5,
        ):
            results.append(future.result())
            elapsed = time.perf_counter() - start
            write_json(
                root / "render_progress.json",
                dict(
                    episodes=len(results),
                    total=len(rows),
                    frames=sum((s["frames"] for s in results)),
                    elapsed_s=elapsed,
                    estimated_remaining_s=elapsed / len(results) * (len(rows) - len(results)),
                ),
            )
    config.update(status="render_complete", render_elapsed_s=time.perf_counter() - start)
    write_json(root / "config.json", config)
    write_json(
        root / "render_summary.json",
        dict(
            episodes=len(results),
            frames=sum((s["frames"] for s in results)),
            bytes=sum((s["bytes"] for s in results)),
            elapsed_s=time.perf_counter() - start,
        ),
    )


if __name__ == "__main__":
    main()
