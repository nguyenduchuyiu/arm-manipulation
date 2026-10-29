"""Render isolated 2 cm voxel depth with pinhole rays, then back-project pixels."""

from sam3_pick.paths import OUTPUTS, DATASETS
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from tqdm import tqdm

DATA = DATASETS / "reach_voxel_grid/ee_delta_25hz_h25_s2"
REFERENCE = OUTPUTS / "e118_reach/metadata"
ROOT = OUTPUTS / "e118_reach"
SIDE = 0.02
POINTS = 128


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def camera():
    base = np.array(json.loads((REFERENCE / "config.json").read_text())["base_origin_world_m"])
    position = base + [0, -1.2, 0.95]
    forward = base + [0, 0, 0.25] - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward), axis=1)
    height, width = (480, 640)
    focal = height / (2 * np.tan(np.deg2rad(70) / 2))
    return dict(
        position=position,
        rotation=rotation,
        base=base,
        height=height,
        width=width,
        K=np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1.0]]),
    )


def voxel_depth(center, cam, side=SIDE):
    signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * 3)).reshape(3, -1).T
    corners = center + signs * side / 2
    projected = (corners - cam["position"]) @ cam["rotation"]
    assert (projected[:, 2] > 0).all(), "Voxel behind camera"
    pixels = projected[:, :2] / projected[:, 2:] * cam["K"][0, 0] + cam["K"][:2, 2]
    assert (pixels.min(0) >= 0).all() and (pixels.max(0) < [cam["width"], cam["height"]]).all(), (
        "Voxel outside camera"
    )
    lo = np.floor(pixels.min(0)).astype(int)
    hi = np.ceil(pixels.max(0)).astype(int)
    u, v = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    uv = np.stack((u.ravel(), v.ravel()), -1)
    rays = np.c_[(uv - cam["K"][:2, 2]) / cam["K"][0, 0], np.ones(len(uv))]
    directions = rays @ cam["rotation"].T
    with np.errstate(divide="ignore", invalid="raise"):
        a = (center - side / 2 - cam["position"]) / directions
        b = (center + side / 2 - cam["position"]) / directions
    near = np.minimum(a, b).max(1)
    far = np.maximum(a, b).min(1)
    valid = (near > 0) & (far >= near)
    assert valid.any(), "Voxel has no depth pixels"
    uv, rays, depth = (uv[valid], rays[valid], near[valid])
    units = np.rint(depth / 0.002).astype(np.uint16)
    recovered = rays * (units.astype(np.float64) * 0.002)[:, None]
    cloud = recovered @ cam["rotation"].T + cam["position"] - cam["base"]
    return (
        cloud.astype(np.float32),
        uv,
        units,
        float(np.max(np.abs(units * 0.002 - depth))),
    )


def generate(output):
    output.mkdir(parents=True, exist_ok=False)
    cam = camera()
    centers = np.load(DATA / "target_world_m.npy")
    clouds = np.empty((len(centers), POINTS, 3), np.float32)
    pixels = np.empty(len(centers), np.int32)
    max_error = 0.0
    started = time.perf_counter()
    rng = np.random.default_rng(118)
    sample_ids = np.linspace(0, len(centers) - 1, 6, dtype=int)
    samples = {}
    for i, center in enumerate(tqdm(centers, desc="Voxel depth -> target cloud", mininterval=5)):
        cloud, uv, units, error = voxel_depth(center, cam)
        max_error = max(max_error, error)
        pixels[i] = len(cloud)
        ids = rng.choice(len(cloud), POINTS, replace=len(cloud) < POINTS)
        clouds[i] = cloud[ids]
        if i in sample_ids:
            samples[f"uv_{i}"] = uv
            samples[f"depth_uint16_{i}"] = units
    np.save(output / "clouds.npy", clouds)
    np.save(output / "visible_pixels.npy", pixels)
    np.savez_compressed(output / "depth_samples.npz", **samples, trajectory_ids=sample_ids)
    np.savez(
        output / "calibration.npz",
        **{k: v for k, v in cam.items() if isinstance(v, np.ndarray)},
    )
    config = dict(
        status="complete",
        trajectories=len(clouds),
        points=POINTS,
        cube_side_m=SIDE,
        input="visible isolated-voxel optical-axis depth + target mask; back-project to robot base",
        renderer="analytic pinhole ray/AABB intersection, equivalent ideal box depth; no robot/table occlusion",
        camera_position_world_m=cam["position"].tolist(),
        camera_rotation_cv_to_world=cam["rotation"].tolist(),
        resolution=[cam["width"], cam["height"]],
        K=cam["K"].tolist(),
        fovy_deg=70,
        depth_unit_m=0.002,
        max_depth_quantization_error_m=max_error,
        valid_pixels_min=int(pixels.min()),
        valid_pixels_median=float(np.median(pixels)),
        valid_pixels_max=int(pixels.max()),
        sampling="128 deterministic random pixels; replacement only if fewer",
        target_center_policy_input=False,
        RGB_policy_input=False,
        centroid_subtraction=False,
        base_origin_world_m=cam["base"].tolist(),
        cloud_bytes=clouds.nbytes,
        source_targets_sha256=sha(DATA / "target_world_m.npy"),
        clouds_sha256=sha(output / "clouds.npy"),
        source_sha256=sha(Path(__file__)),
        elapsed_s=time.perf_counter() - started,
    )
    write(output / "config.json", config)
    (output / "README.md").write_text(
        "# E118 voxel target clouds\n\nComplete ideal single-camera depth export.\n32,017 isolated 2 cm cubes, fixed calibrated wide front view; no scene occlusion/no RGB.\nDepth quantized to 2 mm, then back-projected; 128 base-frame points per goal.\nRaw sample depth pixels and calibration retained. No policy training in this folder.\nExact counts, camera and hashes: config.json.\n"
    )
    print(json.dumps(config), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "clouds")
    generate(parser.parse_args().output)
