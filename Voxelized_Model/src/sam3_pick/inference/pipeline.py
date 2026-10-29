"""E133 preparation and rollout: text SAM3 front/left clouds -> frozen E131."""

from sam3_pick.paths import PICK_DATA as DATA
from sam3_pick.utils.hashing import source_hashes
from sam3_pick.paths import PROJECT, OUTPUTS
import argparse
import gc
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
from tqdm import tqdm
from sam3_pick.simulation.scene import NAMES, BASE
from sam3_pick.utils.depth_image import depth_image
from sam3_pick.models.pointcloud import PointCloudReach
from sam3_pick.simulation.render import visual_model
from sam3_pick.data.rendering import clip_planes
from sam3_pick.data.pick_labels import write
from sam3_pick.evaluation.summary import summarize
from sam3_pick.utils.hashing import sha256_file as sha
from sam3_pick.utils.hashing import state_hash as state_hash
from sam3_pick.data.target_cloud import ROOT as E131, CACHE, extract_cloud
from sam3_pick.inference.rollout import rollout
from sam3_pick.inference.fusion import (
    Rig,
    Observation,
    VIEWS,
    LEFT,
    LOOK_AT,
    VOXEL,
    write_scene,
    filtered_points,
    fuse,
)
from sam3_pick.inference.prompts import PROMPTS, REVISION

ROOT = E131.parent / "e133_sam3"
NOTE = OUTPUTS / "notes/e133_README.md"
CHECKPOINT = E131 / "train/final.pt"
SUITES = dict(
    heldout=(E131 / "test_manifest.jsonl", DATA, 1408),
    xy=(CACHE / "test_xy/manifest.jsonl", CACHE / "test_xy/states", 16),
    ood=(E131 / "test_ood/manifest.jsonl", E131 / "test_ood/states", 394),
)


def load_policy():
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
    assert saved["step"] == 23520
    policy = PointCloudReach().cuda()
    policy.load_state_dict(saved["model"], strict=True)
    policy.eval().requires_grad_(False)
    return policy


def surface_distance(points, center, rotation, half):
    q = np.abs((points - center) @ rotation) - half
    return np.abs(np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0))


def prepare():
    ROOT.mkdir(exist_ok=False)
    shutil.copyfile(NOTE, ROOT / "README.md")
    torch.set_num_threads(4)
    original_hash = sha(DATA / "scene.xml")
    checkpoint_hash = sha(CHECKPOINT)
    assert (
        checkpoint_hash
        == json.loads((E131 / "test_k15/config.json").read_text())["checkpoint_sha256"]
    )
    write_scene(ROOT / "scene.xml")
    rig = Rig(ROOT / "scene.xml")
    config = dict(
        status="prepared_waiting_SAM3_access",
        checkpoint=str(CHECKPOINT),
        checkpoint_sha256=checkpoint_hash,
        step=23520,
        model_revision=REVISION,
        prompts=PROMPTS,
        scene_sha256=sha(ROOT / "scene.xml"),
        source_scene_sha256=original_hash,
        cameras=dict(
            front="unchanged E131 front camera",
            left_world_m=LEFT.tolist(),
            look_at_world_m=LOOK_AT.tolist(),
            projection="pinhole",
            resolution=[320, 240],
            depth_unit_m=0.002,
            depth="ideal optical Z;no added sensor noise",
        ),
        voxel_size_m=VOXEL,
        policy_points=128,
        per_cloud_centering=False,
        training=False,
        architecture_changes_to_E131=False,
        segmentation="SAM3 BF16 RGB+text;top confidence instance>=0.5,mask threshold0.5",
        unavailable_view="empty contribution;record one-view versus two-view observations",
        missing_target="fewer8fused voxels from both predictions:explicit episode failure;no GT or stale-cloud replacement",
        H=25,
        K=15,
        flow_steps=10,
        noise_seed=0,
        frame_budget=300,
        policy_hz=25,
        servo_hz=50,
        physics_hz=500,
        video=False,
        SAM3_inference_run=False,
        suites={},
        source_sha256=source_hashes(),
        sam3_runtime_sha256=sha(PROJECT / "src/sam3_pick/inference/segmentation.py"),
    )
    for suite, (manifest, source, expected) in SUITES.items():
        rows = list(map(json.loads, manifest.read_text().splitlines()))
        assert len(rows) == expected
        assert all(((source / r["path"] / "sim_state.npz").exists() for r in rows))
        shutil.copyfile(manifest, ROOT / f"{suite}_manifest.jsonl")
        config["suites"][suite] = dict(
            cases=expected, manifest_sha256=sha(manifest), source_states=str(source)
        )
    write(ROOT / "config.json", config)
    rows = list(map(json.loads, (ROOT / "heldout_manifest.jsonl").read_text().splitlines()))[:4]
    first = rows[0]
    with np.load(DATA / first["path"] / "sim_state.npz") as z:
        qpos = z["initial_qpos"].copy()
    captured = rig.capture(qpos)
    original = visual_model(
        DATA / "scene.xml", object_bodies=NAMES, lighting_scene=DATA / "scene.xml"
    )
    clip_planes(original)
    original.vis.global_.offwidth = 320
    original.vis.global_.offheight = 240
    state = mujoco.MjData(original)
    state.qpos[:] = qpos
    mujoco.mj_forward(original, state)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0
    with mujoco.Renderer(original, height=240, width=320) as renderer:
        renderer.update_scene(state, camera="cam_front", scene_option=option)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        d = renderer.render().copy()
        valid = np.isfinite(d) & (d > 0) & (d <= 6.5535)
        codes = np.zeros(d.shape, np.uint16)
        codes[valid] = np.rint(d[valid] / 0.002).astype(np.uint16)
    np.testing.assert_array_equal(rgb, captured[0]["rgb"])
    np.testing.assert_array_equal(codes, captured[0]["depth"])
    del original, state
    gc.collect()
    msaa = rig.model.vis.quality.offsamples
    rig.model.vis.quality.offsamples = 0
    seg = mujoco.Renderer(rig.model, height=240, width=320)
    seg.enable_segmentation_rendering()
    rig.model.vis.quality.offsamples = msaa

    def oracle_segment(images, targets):
        masks = []
        meta = []
        for view, target in zip(VIEWS, targets):
            seg.update_scene(rig.state, camera=view, scene_option=rig.option)
            labels = seg.render()
            ids = np.flatnonzero(rig.model.geom_bodyid == rig.model.body(NAMES[target]).id)
            mask = np.isin(labels[..., 0], ids) & (labels[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            masks.append(mask)
            meta.append(
                dict(
                    provider="GT_GEOMETRY_CHECK_ONLY",
                    detections=int(mask.any()),
                    confidence=None,
                )
            )
        return (masks, meta)

    errors = []
    roundtrip = 0.0
    front_parity = []
    K = []
    T = []
    with seg:
        for target in range(4):
            masks, _ = oracle_segment([v["rgb"] for v in captured], [target, target])
            for v, mask in zip(captured, masks):
                points = filtered_points(v["depth"], mask, v["K"], v["T"])
                assert len(points) >= 8
                expected, good, _ = extract_cloud(
                    v["depth"], mask, v["K"], v["T"], np.random.default_rng(42)
                )
                actual = points[
                    np.random.default_rng(42).choice(len(points), 128, replace=len(points) < 128)
                ]
                np.testing.assert_array_equal(expected, actual)
                front_parity.append(True)
                body = rig.model.body(NAMES[target]).id
                half = rig.model.geom(NAMES[target] + "_visual").size
                center = rig.state.xpos[body] - BASE
                rotation = rig.state.xmat[body].reshape(3, 3)
                errors.append(surface_distance(points, center, rotation, half))
                local = (points - v["T"][:3, 3]) @ v["T"][:3, :3]
                uv = local @ v["K"].T
                uv = uv[:, :2] / uv[:, 2:]
                rays = np.c_[
                    (uv[:, 0] - v["K"][0, 2]) * local[:, 2] / v["K"][0, 0],
                    (uv[:, 1] - v["K"][1, 2]) * local[:, 2] / v["K"][1, 1],
                    local[:, 2],
                ]
                restored = rays @ v["T"][:3, :3].T + v["T"][:3, 3]
                roundtrip = max(roundtrip, float(abs(restored - points).max()))
        synthetic = np.c_[np.arange(10) * 0.006, np.zeros(10), np.ones(10) * 0.02].astype(
            np.float32
        )
        cloud, good, stats = fuse([synthetic, synthetic.copy()], np.random.default_rng(0))
        assert good and stats["both_view_voxels"] == 10
        cloud_one, good, stats_one = fuse(
            [synthetic, np.empty((0, 3), np.float32)], np.random.default_rng(0)
        )
        assert good and stats_one["usable_views"] == [True, False]
        np.testing.assert_allclose(cloud, cloud_one, rtol=0, atol=1e-08)
        _, good, stats_empty = fuse([np.empty((0, 3), np.float32)] * 2, np.random.default_rng(0))
        assert not good
        geometry_errors = np.concatenate(errors)
        assert np.percentile(geometry_errors, 95) < 0.003 and roundtrip < 1e-06
        policy = load_policy()
        before = state_hash(policy)
        physical = mujoco.MjModel.from_xml_path(str(DATA / "scene.xml"))
        with TemporaryDirectory(prefix="e133_oracle_plumbing_") as tmp:
            path = Path(tmp)
            (path / "README.md").write_text(
                "Disposable GT-mask geometry/controller compatibility check only;NOT a SAM3 result.\n"
            )
            smoke = rollout(
                rows,
                policy,
                physical,
                Observation(rig, oracle_segment),
                path,
                DATA,
                frame_budget=20,
            )
            assert all((r["frames"] == 20 and r["reason"] == "timeout" for r in smoke))
        assert state_hash(policy) == before
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    for v in captured:
        K.append(v["K"])
        T.append(v["T"])
    np.savez_compressed(
        ROOT / "calibration.npz",
        views=np.array(["front", "left"]),
        intrinsics=np.stack(K),
        camera_to_base=np.stack(T),
        base_origin_world_m=BASE,
        depth_unit_m=0.002,
        resolution=[320, 240],
        projection_model=np.array(["pinhole", "pinhole"]),
    )
    sheet = Image.new("RGB", (1280, 600), "#f4f6f9")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (12, 8),
        "E133 camera preview / GT geometry checks only / SAM3 NOT RUN",
        fill="#21354a",
    )
    for j, (name, v) in enumerate(zip(["Front", "Left (-world X)"], captured)):
        x = j * 640
        draw.text((x + 10, 36), name + " RGB", fill="#21354a")
        sheet.paste(Image.fromarray(v["rgb"]).resize((640, 480)), (x, 60))
    sheet.save(ROOT / "camera_preview.png")
    depth_sheet = Image.new("RGB", (1280, 550), "#f4f6f9")
    draw = ImageDraw.Draw(depth_sheet)
    for j, (name, v) in enumerate(zip(["Front", "Left"], captured)):
        x = j * 640
        draw.text(
            (x + 12, 9),
            name + " optical-Z depth / shared 0.25-1.25m colors",
            fill="#21354a",
        )
        depth_sheet.paste(depth_image(v["depth"]), (x, 45))
    depth_sheet.save(ROOT / "depth_preview.png")
    rig.close()
    assert sha(DATA / "scene.xml") == original_hash and sha(CHECKPOINT) == checkpoint_hash
    verification = dict(
        status="passed",
        SAM3_inference_run=False,
        SAM3_access_pending=True,
        front_RGB_and_depth_unchanged=True,
        both_views_filtered_cloud_parity_with_E131=True,
        GT_points_surface_MAE_mm=float(geometry_errors.mean() * 1000),
        GT_points_surface_P95_mm=float(np.percentile(geometry_errors, 95) * 1000),
        rigid_backprojection_roundtrip_max_error_m=roundtrip,
        synthetic_fusion_pass=True,
        one_empty_view_behavior_pass=True,
        both_empty_views_fail_explicitly=True,
        frozen_E131_20frame_cases=smoke,
        E131_weights_unchanged=True,
        original_dataset_scene_unchanged=True,
    )
    write(ROOT / "preparation_verification.json", verification)
    text = NOTE.read_text().replace(
        "Status: PLANNED.", "Status: PREPARED; SAM3 weight access pending."
    )
    text += "\nGeometry/controller preparation passed. SAM3 inference and all learned-mask rollouts remain NOT RUN. Exact checks: preparation_verification.json.\n"
    NOTE.write_text(text)
    (ROOT / "README.md").write_text(text)
    print(json.dumps(verification), flush=True)


def suite_run(suite, policy, rig, segment, resume):
    source_manifest, source_states, expected = SUITES[suite]
    out = ROOT / f"test_{suite}"
    existing = out.exists()
    if existing:
        assert resume, "Existing suite requires --resume"
    else:
        out.mkdir()
        shutil.copyfile(NOTE, out / "README.md")
    root_cfg = json.loads((ROOT / "config.json").read_text())
    assert (
        sha(CHECKPOINT) == root_cfg["checkpoint_sha256"]
        and sha(ROOT / "scene.xml") == root_cfg["scene_sha256"]
    )
    rows = list(map(json.loads, (ROOT / f"{suite}_manifest.jsonl").read_text().splitlines()))
    assert len(rows) == expected
    results = []
    if existing:
        old = json.loads((out / "config.json").read_text())
        assert (
            old["checkpoint_sha256"] == root_cfg["checkpoint_sha256"]
            and old["sam3_weights_sha256"] == segment.weights_sha256
        )
        assert old["manifest_sha256"] == sha(ROOT / f"{suite}_manifest.jsonl")
        if old["status"] == "complete":
            print(f"Skip already complete suite: {suite}", flush=True)
            return json.loads((out / "summary.json").read_text())
        if (out / "results.jsonl").exists():
            results = list(map(json.loads, (out / "results.jsonl").read_text().splitlines()))
        assert [r["path"] for r in results] == [r["path"] for r in rows[: len(results)]]
    cfg = dict(
        root_cfg,
        status="running",
        suite=suite,
        cases=expected,
        source_states=str(source_states),
        manifest_sha256=sha(ROOT / f"{suite}_manifest.jsonl"),
        sam3_weights=str(segment.weights),
        sam3_weights_sha256=segment.weights_sha256,
        SAM3_inference_run=True,
        GT_masks_as_policy_input=False,
        segmentation_image_batch=2,
        policy_batch=32,
        source_sha256_current=source_hashes(),
    )
    write(out / "config.json", cfg)
    physics = mujoco.MjModel.from_xml_path(str(DATA / "scene.xml"))
    observe = Observation(rig, segment)
    started = time.perf_counter()
    initial = len(results)
    initial_calls = segment.calls
    initial_seg_seconds = segment.seconds
    for start in tqdm(
        range(initial, len(rows), 32),
        desc=f"E133 {suite} SAM3+fusion+E131",
        mininterval=10,
    ):
        current = rollout(
            rows[start : start + 32],
            policy,
            physics,
            observe,
            out,
            source_states,
            progress_path=out / "live.json",
        )
        write(
            ROOT / "SAM3_runtime_verification.json",
            dict(
                **segment.verification,
                peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
            ),
        )
        with (out / "results.jsonl").open("a") as f:
            for r in current:
                f.write(json.dumps(r) + "\n")
        results.extend(current)
        summary = summarize(results, time.perf_counter() - started)
        summary.update(
            status="running",
            complete=len(results),
            total=expected,
            SAM3_pair_calls=segment.calls - initial_calls,
            SAM3_mean_pair_seconds=(segment.seconds - initial_seg_seconds)
            / max(1, segment.calls - initial_calls),
            elapsed_this_launch_s=time.perf_counter() - started,
        )
        write(out / "progress.json", summary)
        print(json.dumps(summary), flush=True)
    assert (
        len(results) == expected
        and sha(CHECKPOINT) == root_cfg["checkpoint_sha256"]
        and (sha(segment.weights) == segment.weights_sha256)
    )
    summary = summarize(results, time.perf_counter() - started)
    summary.update(
        status="complete",
        SAM3_pair_calls=segment.calls - initial_calls,
        SAM3_mean_pair_seconds=(segment.seconds - initial_seg_seconds)
        / max(1, segment.calls - initial_calls),
        one_view_observations=sum((r["single_view_observations"] for r in results)),
        two_view_observations=sum((r["dual_view_observations"] for r in results)),
    )
    if suite == "ood":
        from sam3_pick.evaluation.ood import group_summary

        summary["distributions"] = group_summary(results, rows)
    write(out / "summary.json", summary)
    write(out / "progress.json", summary)
    cfg["status"] = "complete"
    write(out / "config.json", cfg)
    write(
        out / "verification.json",
        dict(
            status="passed",
            all_cases_evaluated=True,
            E131_checkpoint_unchanged=True,
            SAM3_weights_unchanged=True,
            GT_masks_as_policy_input=False,
            no_training=True,
        ),
    )
    with (out / "README.md").open("a") as f:
        f.write("\nCompleted SAM3-predicted-mask rollout:\n" + json.dumps(summary, indent=2) + "\n")
    with (OUTPUTS / "notes/REPORT.md").open("a") as f:
        f.write(f"\nE133 {suite} complete:" + json.dumps(summary) + "\n")
    return summary


@torch.inference_mode()
def run(suite, resume):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = json.loads((ROOT / "config.json").read_text())
    assert sha(CHECKPOINT) == cfg["checkpoint_sha256"]
    assert json.loads((ROOT / "preparation_verification.json").read_text())["status"] == "passed"
    from sam3_pick.inference.segmentation import Sam3Masks

    segment = Sam3Masks()
    policy = load_policy()
    rig = Rig(ROOT / "scene.xml")
    cfg.update(
        status="running",
        SAM3_access_pending=False,
        SAM3_inference_run=False,
        sam3_weights=str(segment.weights),
        sam3_weights_sha256=segment.weights_sha256,
    )
    write(ROOT / "config.json", cfg)
    for name in ("xy", "heldout", "ood") if suite == "all" else [suite]:
        suite_run(name, policy, rig, segment, resume)
    rig.close()
    if suite == "all":
        cfg.update(status="complete", SAM3_inference_run=True)
        write(ROOT / "config.json", cfg)
        text = NOTE.read_text().replace(
            "Status: PREPARED; SAM3 weight access pending.", "Status: COMPLETE."
        )
        text += "\nAll requested learned-mask rollout suites complete. See each test suite summary.json.\n"
        NOTE.write_text(text)
        (ROOT / "README.md").write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--suite", choices=["all", *SUITES], default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else run(args.suite, args.resume)
