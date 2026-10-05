"""One continuous 25 Hz reveal, shuffle, and physical pick/place episode."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import imageio.v3 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from controllers.oracle_pick import GraspFailure, IKFailure, pick
from OATFlow.dataset.perturbation import ContinuousPerturbation, RolloutFailure
from OATFlow.task import normalized
from OATFlow.environment.env import (COVER_XY, COVER_DROP_ZONE_XY,
                                  TARGET_DROP_ZONE_XY, TARGETS, MemoryOcclusionEnv, wrist_camera_metadata)

FPS = 25


def make_context_plan(seed, swaps):
    rng = np.random.default_rng(seed + 20000)
    centers = {name: [float(x + rng.uniform(-.008, .008)),
                      float(y + rng.uniform(-.002, .002))]
               for name, (x, y) in COVER_XY.items()}
    paths = []
    for _ in range(swaps):
        endpoints = {name: [float(COVER_XY[other][0] + rng.uniform(-.008, .008)),
                            float(COVER_XY[name][1] + rng.uniform(-.002, .002))]
                     for name, other in (("cover_a", "cover_b"),
                                         ("cover_b", "cover_a"))}
        paths.append({"frames": int(rng.integers(55, 91)),
                      "arc_m": float(rng.uniform(.07, .12)),
                      "speed_gamma": float(rng.uniform(.8, 1.25)),
                      "route_flip": int(rng.choice((-1, 1))),
                      "pause_frames": int(rng.integers(5, 21)),
                      "endpoints": endpoints})
    return {"reveal_frames": int(rng.integers(30, 46)),
            "occlude_frames": int(rng.integers(65, 91)),
            "hold_frames": int(rng.integers(20, 36)),
            "initial_centers": centers, "swaps": paths}


def context(env, plan, record):
    """Scene setup is scripted; covered objects translate with their covers."""
    def frame(phase):
        mujoco.mj_forward(env.model, env.data)
        record(phase, False)
        env.data.time += 1 / FPS

    for _ in range(plan["reveal_frames"]):
        frame("reveal")
    for i in range(plan["occlude_frames"]):
        t = (i + 1) / plan["occlude_frames"]
        travel, lower = min(t / .6, 1), max((t - .6) / .4, 0)
        travel, lower = travel**2 * (3 - 2 * travel), lower**2 * (3 - 2 * lower)
        for index, (name, address) in enumerate(env.cover_qadr.items()):
            x, y = plan["initial_centers"][name]
            start = -.15 if index == 0 else 1.25
            env.data.qpos[address:address + 3] = (start + travel * (x - start), y,
                                                 .22 + lower * (.002 - .22))
        frame("occlude")
    for swap in plan["swaps"]:
        starts = {name: env.data.qpos[address:address + 3].copy()
                  for name, address in env.cover_qadr.items()}
        object_starts = env.target_positions()
        mid_x = (starts["cover_a"][0] + starts["cover_b"][0]) / 2
        for i in range(swap["frames"]):
            u = ((i + 1) / swap["frames"]) ** swap["speed_gamma"]
            travel = u * u * (3 - 2 * u)
            for name, address in env.cover_qadr.items():
                end = np.asarray(swap["endpoints"][name])
                side = np.sign(starts[name][0] - mid_x) * swap["route_flip"]
                xy = starts[name][:2] + travel * (end - starts[name][:2])
                xy[1] += side * swap["arc_m"] * np.sin(np.pi * travel)
                position = np.array((xy[0], xy[1], .002))
                env.data.qpos[address:address + 3] = position
                for target, owner in env.assignment.items():
                    if owner == name:
                        adr = env.target_qadr[target]
                        env.data.qpos[adr:adr + 3] = object_starts[target] + position - starts[name]
            frame("shuffle")
        for _ in range(swap["pause_frames"]):
            frame("shuffle")
    for _ in range(plan["hold_frames"]):
        frame("hold")
    env.initial_positions = env.target_positions()
    env.phase = "occlude"


def execute(env, target, callback, perturbation=None):
    env.query(target)
    cover = env.assignment[target]
    other_cover = next(name for name in env.cover_qadr if name != cover)
    other_start = env.data.xpos[env.model.body(other_cover).id].copy()
    env.choose_cover(cover)

    def checked(stage):
        env._evaluate()
        if env.failure_reason:
            raise RolloutFailure(env.failure_reason)
        callback(stage)

    def attempt(key, body, destination, straight_lift=False):
        attempts = perturbation.config["max_attempts"] if perturbation else 1
        for number in range(1, attempts + 1):
            point = env.data.xpos[env.model.body(body).id].copy()
            if key == "cover":
                point += (env.data.xmat[env.model.body(body).id].reshape(3, 3) @ (0., 0., .130)
                          if perturbation else np.array((0., 0., .130)))
            else:
                point[2] += .015 if body == "Milk" else .010
            if perturbation:
                perturbation.scale = 1.0 if number == 1 else perturbation.config["retry_scale"]
            event = {"body": body, "stage": key, "attempt": number,
                     "start_frame": len(perturbation.trace) + perturbation.start_frame} if perturbation else {}
            try:
                gain = pick(env.model, env.data, body, point, place_xy=destination,
                            lift_height=.20, straight_lift=straight_lift,
                            on_step=lambda stage: checked(key + "_" + stage),
                            perturb_command=(lambda stage, command: perturbation.apply(
                                key + "_" + stage, command,
                                perturbation.start_frame + len(perturbation.trace))) if perturbation else None)
            except GraspFailure as error:
                if not perturbation:
                    raise
                event.update(success=False, reason=str(error),
                             end_frame=perturbation.start_frame + len(perturbation.trace))
                perturbation.attempts.append(event)
                if number == attempts:
                    raise RolloutFailure(str(error)) from error
                continue
            if perturbation:
                deposited = env._cover_in_drop_zone() if key == "cover" else env._target_in_drop_zone()
                if not deposited:
                    event.update(success=False, lift_gain_m=gain, reason="not deposited in drop zone",
                                 end_frame=perturbation.start_frame + len(perturbation.trace))
                    perturbation.attempts.append(event)
                    if number == attempts:
                        raise RolloutFailure(f"{body}: not deposited after {attempts} attempts")
                    continue
                event.update(success=True, lift_gain_m=gain,
                             end_frame=perturbation.start_frame + len(perturbation.trace))
                perturbation.attempts.append(event)
            return gain

    cover_gain = attempt("cover", cover, COVER_DROP_ZONE_XY, straight_lift=True)
    if not env._cover_in_drop_zone():
        raise RolloutFailure("cover was not deposited in its drop zone")
    callback("t_obj")
    target_gain = attempt("object", target, TARGET_DROP_ZONE_XY)
    if not env._target_in_drop_zone():
        raise RolloutFailure("target was not deposited in its drop zone")
    if np.linalg.norm(env.data.xpos[env.model.body(other_cover).id][:2] - other_start[:2]) > .01:
        raise RolloutFailure("expert displaced the other cover")
    env._evaluate()
    if not env.success:
        raise RolloutFailure("target was not held long enough before placement")
    return cover_gain, target_gain


def generate(root: Path, seed: int, target: str, swaps: int = 1, previews: bool = True,
             layout: str = "standard", tcow_labels: bool = False,
             control_perturbation=None, directory: Path | None = None):
    if target not in TARGETS or swaps not in (1, 2, 3):
        raise ValueError("use a HOPE target and one to three swaps")
    directory = directory if directory is not None else root / f"episode_{seed:06d}_{target}"
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("episode.json", "input.json"):
        (directory / filename).unlink(missing_ok=True)
    env = MemoryOcclusionEnv(context_fps=25, resolution=320)
    env.reset(seed=seed, options={"upright_targets": True, "layout": layout})
    context_plan = make_context_plan(seed, swaps)
    env.data.time = 0.0
    cover = env.assignment[target]
    cover_geoms = [i for i in range(env.model.ngeom)
                   if env.model.geom(i).name.startswith(cover + "_")]
    target_geom = env.model.geom(target + "_visual").id
    if tcow_labels:
        original_target_group = int(env.model.geom_group[target_geom])
        original_cover_groups = env.model.geom_group[cover_geoms].copy()
        target_only = mujoco.MjvOption()
        target_only.geomgroup[:] = 0
        target_only.geomgroup[5] = 1
        cover_only = mujoco.MjvOption()
        cover_only.geomgroup[:] = 0
        cover_only.geomgroup[4] = 1
    top = 40 if tcow_labels else 0
    height = 240 if tcow_labels else 320
    masks, entities, proprio, actions, valid, phases, poses, times = [], [], [], [], [], [], [], []
    oracle_actions, executed_actions = [], []
    tcow_masks = []
    qpos_frames = []
    limits = env.model.actuator_ctrlrange.copy()
    decision = {}
    grasp = {}
    grasp_streak = {"cover": 0, "object": 0}
    jaw_geoms = {env.model.geom(f"link_6_{side}_jaw_collision_0").id
                 for side in ("left", "right")}
    grasp_start_z = {}
    semantic_entity = 2  # target until it is fully hidden
    transitions = {}
    perturbation = (ContinuousPerturbation(control_perturbation, limits)
                    if control_perturbation is not None else None)
    def video_writer(name):
        writer = iio.imopen(directory / name, "w", plugin="pyav")
        writer.init_video_stream("libx264", fps=FPS, pixel_format="yuv420p")
        return writer

    rgb_writer = video_writer("rgb.mp4")
    wrist_writer = video_writer("wrist_rgb.mp4")
    mask_writer = video_writer("mask_preview.mp4") if previews else None
    progress = tqdm(desc=f"collect {directory.name}", unit="frame", mininterval=5, file=sys.stdout)

    def record(phase, action_valid=True):
        nonlocal semantic_entity
        # All saved poses and both camera views refer to the same current state.
        mujoco.mj_forward(env.model, env.data)
        rgb_full = env._overview()
        seg_full = env.segmentation()
        target_visible_full = seg_full[:, :, 0] == target_geom
        index = len(times)
        if phase == "occlude" and semantic_entity == 2 and not target_visible_full.any():
            semantic_entity = 1
            transitions["fully_occluded"] = index
        if phase in ("cover_retreat", "cover_home") and semantic_entity == 1 and (
                target_visible_full.any() and env._cover_in_drop_zone()):
            semantic_entity = 2
            transitions["cover_released"] = index
        if phase in ("object_retreat", "object_home") and semantic_entity == 2 and env._target_in_drop_zone():
            semantic_entity = 0
            transitions["object_released"] = index
        if phase in ("shuffle", "hold"):
            ids = [env.model.geom(name + "_visual").id for name in TARGETS]
            if np.any(np.isin(seg_full[:, :, 0], ids)):
                raise RuntimeError("an object became visible during shuffle/hold")
        rgb = rgb_full[top:top + height]
        seg = seg_full[top:top + height]
        target_visible = target_visible_full[top:top + height]
        mask = np.zeros((height, 320), dtype=np.uint8)
        if semantic_entity == 1:
            mask[np.isin(seg[:, :, 0], cover_geoms)] = 1
        elif semantic_entity == 2:
            mask[target_visible] = 2
        if tcow_labels:
            env.model.geom_group[target_geom] = 5
            env.model.geom_group[cover_geoms] = 4
            env.segmentation_renderer.update_scene(env.data, camera="overview", scene_option=target_only)
            target_amodal = env.segmentation_renderer.render()[top:top + height, :, 0] == target_geom
            env.segmentation_renderer.update_scene(env.data, camera="overview", scene_option=cover_only)
            cover_amodal = np.isin(env.segmentation_renderer.render()[top:top + height, :, 0], cover_geoms)
            env.model.geom_group[target_geom] = original_target_group
            env.model.geom_group[cover_geoms] = original_cover_groups
            covered_pixels = (target_amodal & np.isin(seg[:, :, 0], cover_geoms)).sum()
            occlusion_fraction = 1 - target_visible.sum() / max(target_amodal.sum(), 1)
            frontmost_cover = occlusion_fraction >= .95 and covered_pixels / max(target_amodal.sum(), 1) >= .475
            cover_body = env.model.body(cover).id
            target_body = env.model.body(target).id
            local = env.data.xmat[cover_body].reshape(3, 3).T @ (
                env.data.xpos[target_body] - env.data.xpos[cover_body])
            half_height = env.model.geom(target + "_collision").size[2]
            contained = (abs(local[0]) < .073 and abs(local[1]) < .058
                         and local[2] - half_height >= -.005 and local[2] + half_height <= .068)
            tcow_masks.append(np.stack((target_amodal,
                                        cover_amodal if frontmost_cover else np.zeros_like(cover_amodal),
                                        cover_amodal if contained else np.zeros_like(cover_amodal))))
        rgb_writer.write(rgb, is_batch=False)
        wrist_writer.write(env.wrist_image(), is_batch=False)
        if previews:
            overlay = rgb.copy()
            overlay[mask == 1] = (overlay[mask == 1] * .4 + np.array((30, 100, 255)) * .6).astype(np.uint8)
            overlay[mask == 2] = (overlay[mask == 2] * .4 + np.array((255, 220, 0)) * .6).astype(np.uint8)
            preview = Image.fromarray(overlay)
        label = ("task complete", "occluder | blue", "query object | yellow")[semantic_entity]
        if previews:
            draw = ImageDraw.Draw(preview)
            draw.rectangle((0, 0, 319, 20), fill=(0, 0, 0))
            draw.text((5, 4), f"{index / FPS:.2f}s  {label}", fill=(255, 255, 255))
            mask_writer.write(np.asarray(preview), is_batch=False)
        masks.append(mask)
        entities.append(semantic_entity)
        proprio.append(normalized(env.data.qpos[env.robot_qpos_addresses], limits))
        if perturbation:
            oracle = perturbation.trace[-1]["expert_ctrl"] if action_valid else env.data.ctrl
            oracle_actions.append(normalized(oracle, limits))
            executed_actions.append(normalized(env.data.ctrl, limits))
            actions.append(oracle_actions[-1] if perturbation.config["labels"] == "expert"
                           else executed_actions[-1])
        else:
            actions.append(normalized(env.data.ctrl, limits))
        valid.append(action_valid)
        phases.append(phase)
        times.append(float(env.data.time))
        poses.append(np.concatenate([env.data.qpos[a:a + 7] for a in
                                     [*env.target_qadr.values(), *env.cover_qadr.values()]]))
        qpos_frames.append(env.data.qpos.copy())
        if phase == "reveal" and len(times) == 1:
            Image.fromarray(rgb).save(directory / "reveal.png")
        if phase == "done":
            Image.fromarray(rgb).save(directory / "final.png")
        for name, index in {**decision, **{f"t_{key}_grasp": value
                                             for key, value in grasp.items()}}.items():
            if len(times) - 1 == index:
                Image.fromarray(rgb).save(directory / f"{name}.png")
        progress.set_postfix(phase=phase, refresh=False)
        progress.update(1)

    def securely_lifted(body_name, start_z):
        body_id = env.model.body(body_name).id
        if env.data.xpos[body_id, 2] - start_z < 0.01:
            return False
        touched = set()
        for contact in env.data.contact:
            if env.model.geom_bodyid[contact.geom1] == body_id and contact.geom2 in jaw_geoms:
                touched.add(contact.geom2)
            if env.model.geom_bodyid[contact.geom2] == body_id and contact.geom1 in jaw_geoms:
                touched.add(contact.geom1)
        return touched == jaw_geoms

    try:
        context(env, context_plan, record)
        grasp_start_z["cover"] = float(env.data.xpos[env.model.body(cover).id, 2])
        decision["t_occ"] = len(times)
        if perturbation:
            perturbation.start_frame = len(times)
        ordered = sorted(env.cover_qadr, key=lambda name: env.data.qpos[env.cover_qadr[name]])
        correct_id = ordered.index(cover)

        def callback(stage):
            if stage == "t_obj":
                decision["t_obj"] = len(times)
                grasp_start_z["object"] = float(env.data.xpos[env.model.body(target).id, 2])
            else:
                if stage.endswith("_approach"):
                    grasp_streak[stage.split("_")[0]] = 0
                for key, body_name in (("cover", cover), ("object", target)):
                    if stage in (f"{key}_lift", f"{key}_hold") and key not in grasp:
                        grasp_streak[key] = (grasp_streak[key] + 1 if securely_lifted(
                            body_name, grasp_start_z[key]) else 0)
                        if grasp_streak[key] >= 3:
                            grasp[key] = len(times)
                record(stage)

        failure_reason = None
        gains = (None, None)
        try:
            gains = execute(env, target, callback, perturbation)
        except (RolloutFailure, IKFailure) as error:
            if perturbation is None:
                raise
            failure_reason = str(error)
        if failure_reason is None and set(grasp) != {"cover", "object"}:
            raise RuntimeError(f"missing grasp event: {set(('cover', 'object')) - set(grasp)}")
        if failure_reason is None and set(transitions) != {"fully_occluded", "cover_released", "object_released"}:
            raise RuntimeError(f"missing semantic transition: {transitions}")
        record("done" if failure_reason is None else "failed", False)
        np.savez_compressed(directory / "observation.npz", joint_position=np.stack(proprio), timestamp_s=np.array(times))
        grasp_event = np.zeros(len(times), dtype=np.uint8)
        for key, frame in grasp.items():
            grasp_event[frame] = 1 if key == "cover" else 2
        np.savez_compressed(directory / "supervision.npz", mask=np.stack(masks),
                            expert_action=np.stack(actions), action_valid=np.array(valid),
                            phase=np.array(phases), grasp_event=grasp_event,
                            semantic_entity=np.array(entities, dtype=np.uint8),
                            **({"oracle_action": np.stack(oracle_actions),
                                "executed_action": np.stack(executed_actions)} if perturbation else {}))
        if tcow_labels:
            tcow = np.stack(tcow_masks)
            np.savez_compressed(directory / "tcow_labels.npz", mask=tcow,
                                channel_names=np.array(("target_amodal", "frontmost_occluder", "outermost_container")),
                                timestamp_s=np.array(times))
            Image.fromarray(((masks[0] == 2) * 255).astype(np.uint8)).save(
                directory / "query_mask.png")
        body_order = [*env.target_qadr, *env.cover_qadr]
        np.savez_compressed(directory / "debug_poses.npz", poses=np.stack(poses),
                            body_names=np.array(body_order), timestamp_s=np.array(times),
                            qpos=np.stack(qpos_frames))
        phase_intervals = []
        start = 0
        for end in range(1, len(phases) + 1):
            if end == len(phases) or phases[end] != phases[start]:
                phase_intervals.append({"phase": phases[start], "start_frame": start,
                                        "end_frame_exclusive": end,
                                        "start_s": start / FPS, "end_s": end / FPS})
                start = end
        split = "test" if seed % 10 == 9 else "val" if seed % 10 == 8 else "train"
        metadata = {"seed": seed, "query_rgb": f"../references/{target}.png",
                    "split": split, "layout": env.layout,
                    "object_ids": list(env.target_qadr), "occluder_ids": list(env.cover_qadr),
                    "object_to_occluder": dict(env.assignment),
                    "cover_ids_at_t_occ": ordered,
                    "visual_input": "rgb", "fps": FPS, "frames": len(times), "resolution": [height, 320],
                    "overview_camera": {"position": env.model.camera("overview").pos.tolist(),
                                        "quaternion": env.model.camera("overview").quat.tolist(),
                                        "fovy": float(env.model.camera("overview").fovy[0]),
                                        "crop_top": top},
                    "wrist_camera": wrist_camera_metadata(env.model, len(times)),
                    "tcow_labels": "tcow_labels.npz" if tcow_labels else None,
                    "decision_frames": decision,
                    "decision_times_s": {k: v / FPS for k, v in decision.items()},
                    "grasp_frames": grasp,
                    "semantic_transition_frames": transitions,
                    "semantic_transition_times_s": {k: v / FPS for k, v in transitions.items()},
                    "phase_intervals": phase_intervals,
                    "grasp_times_s": {k: v / FPS for k, v in grasp.items()},
                    "grasp_event_rule": "both jaws contact body, body lifted >=1 cm for 3 frames",
                    "grasp_event_ids": {"0": "none", "1": "cover", "2": "object"},
                    "correct_cover_id": correct_id, "cover_id_semantics": "left=0,right=1 at t_occ",
                    "swaps": swaps, "target_object_id": target, "correct_cover_body": cover,
                    "context_plan": context_plan,
                    "lift_gains_m": gains, "success": failure_reason is None,
                    "mask_ids": {"0": "background", "1": "correct cover", "2": "query target"},
                    "semantic_entity_ids": {"0": "task complete", "1": "responsible cover", "2": "query target"},
                    "semantic_rule": "target until fully occluded; cover until release completes; target until its release completes",
                    "normalization_limits": limits.tolist(), "gripper": "0 closed, 1 open",
                    "joint_names": list(env.robot_joint_names),
                    "context_setup": "scripted covers and contents; expert uses physics contact",
                    "pose_body_order": body_order,
                    "pose_format": "world xyz (m), quaternion wxyz; 7 values per body"}
        if perturbation:
            metadata["control_perturbation"] = {
                **perturbation.config, "start_frame": decision["t_occ"],
                "end_frame_exclusive": len(times) - 1, "trace": "perturbation.npz",
                "oracle_attempts": perturbation.attempts,
                "type": "continuous_correlated_actuator_noise", "failure_reason": failure_reason}
            trace = [row for row in perturbation.trace if row["frame"] < len(times) - 1]
            np.savez_compressed(directory / "perturbation.npz",
                frames=np.array([row["frame"] for row in trace]),
                phase=np.array([row["phase"] for row in trace]),
                expert_ctrl=np.stack([row["expert_ctrl"] for row in trace]),
                executed_ctrl=np.stack([row["executed_ctrl"] for row in trace]),
                noise_scale=np.array([row["scale"] for row in trace]))
        (directory / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (directory / "input.json").write_text(json.dumps({
            "rgb_video": "rgb.mp4", "observation": "observation.npz",
            "wrist_rgb_video": "wrist_rgb.mp4",
            "query_rgb": metadata["query_rgb"], "decision_frames": decision,
        }, indent=2) + "\n")
        return metadata
    finally:
        progress.close()
        rgb_writer.close()
        wrist_writer.close()
        if previews:
            mask_writer.close()
        env.close()
