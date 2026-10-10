"""Continuous context + expert, or expert-only episodes with absolute joint labels."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import imageio.v3 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from controllers.oracle_pick import GraspFailure
from OATFlow.dataset.expert import execute, setup_scene
from OATFlow.dataset.references import create_references
from OATFlow.environment.env import (COVER_XY, COVER_DROP_ZONE_XY, COVER_DROP_ZONE_HALF_SIZE,
                                    TARGETS, TARGET_HALF_HEIGHT, MemoryOcclusionEnv, wrist_camera_metadata)
from OATFlow.environment.success import jaw_contacts
from OATFlow.task import TASKS, normalized, task_id_from_state

FPS = 25


def make_context_plan(seed, swaps, centers):
    rng = np.random.default_rng(seed + 20000)
    paths = []
    for swap_index in range(swaps):
        endpoints = {name: [float(COVER_XY[other if swap_index % 2 == 0 else name][0] + rng.uniform(-.025, .025)),
                            float(COVER_XY[name][1] + rng.uniform(-.025, .025))]
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
    # Scripted shuffle changes positions without physical integration.
    # Discard velocities from the reveal setup before handing control to the expert.
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    env.initial_positions = env.target_positions()
    env.phase = "occlude"


def setup_context_scene(env, task_id, seed, swaps):
    """Use the prior setup, then reveal/shuffle it without resetting identities."""
    initial_task = task_id ^ (swaps % 2)
    rng, _ = setup_scene(env, initial_task, seed)
    centers = {name: env.data.qpos[address:address + 2].tolist()
               for name, address in env.cover_qadr.items()}
    plan = make_context_plan(seed, swaps, centers)
    env._park_covers()
    env.phase = "reveal"
    return rng, plan


def generate(root: Path, seed: int, target: str, swaps: int = 1, previews: bool = False,
             layout: str = "standard", include_context: bool = True,
             task_id: int | None = None, demo_id: int = 0, directory: Path | None = None):
    if target not in TARGETS or swaps not in (1, 2, 3):
        raise ValueError("use a configured target and one to three swaps")
    if task_id is None:
        raise ValueError("collection requires the final covered task ID")
    if target not in TASKS[task_id]["objects"] or layout != "standard":
        raise ValueError("target must belong to the task; prior setup uses standard layout")
    create_references(root)
    directory = directory or root / f"episode_{seed:06d}_{target}"
    directory.mkdir(parents=True)
    env = MemoryOcclusionEnv(resolution=320)
    writers = []
    progress = tqdm(desc=f"collect {directory.name}", unit="frame", mininterval=5, file=sys.stdout)
    try:
        if include_context:
            rng, plan = setup_context_scene(env, task_id, seed, swaps)
        else:
            rng, _ = setup_scene(env, task_id, seed)
            plan = None
        env.data.time = 0
        cover = env.assignment[target]
        cover_geoms = [i for i in range(env.model.ngeom) if env.model.geom(i).name.startswith(cover + "_")]
        target_geom = env.model.geom(target + "_visual").id
        if include_context:
            original_target_group = int(env.model.geom_group[target_geom])
            original_cover_groups = env.model.geom_group[cover_geoms].copy()
            target_only, cover_only = mujoco.MjvOption(), mujoco.MjvOption()
            target_only.geomgroup[:] = 0
            target_only.geomgroup[5] = 1
            cover_only.geomgroup[:] = 0
            cover_only.geomgroup[4] = 1
        elif np.isin(env.segmentation()[:, :, 0],
                     [env.model.geom(name + "_visual").id for name in TARGETS]).any():
            raise GraspFailure("an object is visible in the initial covered state")
        limits = env.model.actuator_ctrlrange.copy()
        proprio, actions, valid, phases, qposes, qvels = [], [], [], [], [], []
        masks, entities, tcow_masks = [], [], []
        decision, grasp, transitions = {}, {}, {}
        grasp_streak = {"cover": 0, "object": 0}
        grasp_start_z = {}
        semantic_entity = 2 if include_context else 1
        for name in ("rgb.mp4", "wrist_rgb.mp4"):
            writer = iio.imopen(directory / name, "w", plugin="pyav")
            writer.init_video_stream("libx264", fps=FPS, pixel_format="yuv420p")
            writers.append(writer)
        preview_writer = None
        if previews and include_context:
            preview_writer = iio.imopen(directory / "mask_preview.mp4", "w", plugin="pyav")
            preview_writer.init_video_stream("libx264", fps=FPS, pixel_format="yuv420p")
            writers.append(preview_writer)

        def record(phase, action_valid=True):
            nonlocal semantic_entity
            index = len(phases)
            if phase == "t_obj":
                decision["t_obj"] = index
                grasp_start_z["object"] = float(env.data.xpos[env.model.body(target).id, 2])
                semantic_entity = 2
                transitions["cover_released"] = index
                return
            mujoco.mj_forward(env.model, env.data)
            rgb_full = env._overview()
            if phase == "done":
                semantic_entity = 0
                transitions["task_complete"] = index
            for key, body in (("cover", cover), ("object", target)):
                if phase == body + "_approach":
                    grasp_streak[key] = 0
                if phase in (body + "_lift", body + "_hold") and key not in grasp:
                    lifted = env.data.xpos[env.model.body(body).id, 2] - grasp_start_z[key] > .01
                    grasp_streak[key] = grasp_streak[key] + 1 if lifted and len(jaw_contacts(env.model, env.data, body)) == 2 else 0
                    if grasp_streak[key] >= 3:
                        grasp[key] = index
            if include_context:
                seg_full = env.segmentation()
                target_visible_full = seg_full[:, :, 0] == target_geom
                if phase == "occlude" and semantic_entity == 2 and not target_visible_full.any():
                    semantic_entity = 1
                    transitions["fully_occluded"] = index
                if phase in ("shuffle", "hold") and np.isin(seg_full[:, :, 0],
                        [env.model.geom(name + "_visual").id for name in TARGETS]).any():
                    raise GraspFailure("an object became visible during shuffle/hold")
                top, height = 40, 240
                seg = seg_full[top:top + height]
                target_visible = target_visible_full[top:top + height]
                mask = np.zeros((height, 320), dtype=np.uint8)
                if semantic_entity == 1:
                    mask[np.isin(seg[:, :, 0], cover_geoms)] = 1
                elif semantic_entity == 2:
                    mask[target_visible] = 2
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
                half_height = TARGET_HALF_HEIGHT[target]
                contained = (abs(local[0]) < .073 and abs(local[1]) < .058
                             and local[2] - half_height >= -.005 and local[2] + half_height <= .068)
                tcow_masks.append(np.stack((target_amodal,
                                            cover_amodal if frontmost_cover else np.zeros_like(cover_amodal),
                                            cover_amodal if contained else np.zeros_like(cover_amodal))))
                masks.append(mask)
                entities.append(semantic_entity)
                if preview_writer is not None:
                    overlay = rgb_full[40:280].copy()
                    for entity, color in ((1, (30, 100, 255)), (2, (255, 220, 0))):
                        overlay[mask == entity] = (overlay[mask == entity] * .4 + np.array(color) * .6).astype(np.uint8)
                    preview = Image.fromarray(overlay)
                    ImageDraw.Draw(preview).text((5, 4), f"{index / FPS:.2f}s {phase}", fill=(255, 255, 255))
                    preview_writer.write(np.asarray(preview), is_batch=False)
            writers[0].write(rgb_full, is_batch=False)
            writers[1].write(env.wrist_image(), is_batch=False)
            proprio.append(normalized(env.data.qpos[env.robot_qpos_addresses], limits))
            actions.append(normalized(env.data.ctrl, limits))
            valid.append(action_valid)
            phases.append(phase)
            qposes.append(env.data.qpos.copy())
            qvels.append(env.data.qvel.copy())
            if index == 0:
                Image.fromarray(rgb_full).save(directory / ("reveal.png" if include_context else "initial.png"))
            if phase == "done":
                Image.fromarray(rgb_full).save(directory / "final.png")
            for name, frame in {**decision, **{f"t_{key}_grasp": value for key, value in grasp.items()}}.items():
                if index == frame:
                    Image.fromarray(rgb_full).save(directory / f"{name}.png")
            progress.update(1)
            progress.set_postfix(phase=phase, refresh=False)

        if include_context:
            context(env, plan, record)
        else:
            record("covered_start", False)
        decision["t_occ"] = len(phases)
        initial_qpos, initial_qvel, initial_ctrl = env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy()
        grasp_start_z["cover"] = float(env.data.xpos[env.model.body(cover).id, 2])
        pair = [name for name in TARGETS if env.assignment[name] == cover]
        actual_task = task_id_from_state(env, pair)
        if task_id is not None and actual_task != task_id:
            raise ValueError("task ID differs from covered scene")
        expert = execute(env, target, rng, record)
        if set(grasp) != {"cover", "object"}:
            raise GraspFailure("missing bilateral grasp event")
        record("done", False)
        n = len(phases)
        timestamps = np.arange(n) / FPS
        np.savez_compressed(directory / "observation.npz", joint_position=np.stack(proprio), timestamp_s=timestamps)
        event = np.zeros(n, dtype=np.uint8)
        for key, frame in grasp.items():
            event[frame] = 1 if key == "cover" else 2
        supervision = dict(expert_action=np.stack(actions), action_valid=np.array(valid),
                           phase=np.array(phases), grasp_event=event)
        if include_context:
            supervision.update(mask=np.stack(masks), semantic_entity=np.array(entities, dtype=np.uint8))
            np.savez_compressed(directory / "tcow_labels.npz", mask=np.stack(tcow_masks),
                                channel_names=np.array(("target_amodal", "frontmost_occluder", "outermost_container")),
                                timestamp_s=timestamps)
            Image.fromarray(((masks[0] == 2) * 255).astype(np.uint8)).save(directory / "query_mask.png")
        np.savez_compressed(directory / "supervision.npz", **supervision)
        np.savez_compressed(directory / "sim_state.npz", initial_qpos=initial_qpos, initial_qvel=initial_qvel,
                            initial_ctrl=initial_ctrl, qpos=np.stack(qposes), qvel=np.stack(qvels))
        # Cover IDs refer to the actionable scene, before the expert moves the cover.
        ordered = sorted(env.cover_qadr, key=lambda name: initial_qpos[env.cover_qadr[name]])
        intervals, start = [], 0
        for end in range(1, n + 1):
            if end == n or phases[end] != phases[start]:
                intervals.append(dict(phase=phases[start], start_frame=start, end_frame_exclusive=end,
                                      start_s=start / FPS, end_s=end / FPS))
                start = end
        split = ("test" if seed % 10 == 9 else "val" if seed % 10 == 8 else "train") if include_context else "train"
        metadata = dict(seed=seed, demo_id=demo_id, task_id=actual_task, target_id=TARGETS.index(target),
                        setup_task_id=task_id ^ (swaps % 2) if include_context else task_id,
                        task={"objects": pair, "cover_side": ordered.index(cover)},
                        split=split, layout=layout, collection_mode="context" if include_context else "expert",
                        context_plan=plan, swaps=swaps if include_context else 0,
                        initial_phase="reveal" if include_context else "occlude",
                        frames=n, fps=FPS, resolution=[320, 320], visual_input="rgb",
                        tcow_resolution=[240, 320] if include_context else None,
                        tcow_labels="tcow_labels.npz" if include_context else None,
                        query_rgb=os.path.relpath(root / "references" / f"{target}.png", directory),
                        object_ids=list(env.target_qadr), occluder_ids=list(env.cover_qadr),
                        object_to_occluder=dict(env.assignment), cover_ids_at_t_occ=ordered,
                        correct_cover_id=ordered.index(cover), correct_cover_body=cover,
                        target_object_id=target, decision_frames=decision,
                        decision_times_s={k: v / FPS for k, v in decision.items()},
                        grasp_frames=grasp, grasp_times_s={k: v / FPS for k, v in grasp.items()},
                        grasp_event_rule="both jaws contact body, lifted >=1 cm for three frames",
                        semantic_transition_frames=transitions,
                        semantic_transition_times_s={k: v / FPS for k, v in transitions.items()},
                        phase_intervals=intervals, normalization_limits=limits.tolist(),
                        joint_names=list(env.robot_joint_names), gripper="absolute 0 closed, 1 open",
                        action_representation="absolute_joint", expert_profile="fast",
                        objective="cover_drop_then_target_lift", success=True, target_grasped=True,
                        target_lifted=True, cover_deposited=True,
                        cover_drop_zone_xy=list(COVER_DROP_ZONE_XY), cover_drop_zone_half_size=list(COVER_DROP_ZONE_HALF_SIZE),
                        cover_release_height_m=.12,
                        final_target_position=env.data.xpos[env.model.body(target).id].tolist(),
                        overview_camera={"position": env.model.camera("overview").pos.tolist(),
                                         "quaternion": env.model.camera("overview").quat.tolist(),
                                         "fovy": float(env.model.camera("overview").fovy[0]), "crop_top": 40},
                        wrist_camera=wrist_camera_metadata(env.model, n), **expert)
        (directory / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (directory / "input.json").write_text(json.dumps(dict(rgb_video="rgb.mp4", wrist_rgb_video="wrist_rgb.mp4",
                                    observation="observation.npz", query_rgb=metadata["query_rgb"], decision_frames=decision), indent=2) + "\n")
        return metadata
    finally:
        progress.close()
        for writer in writers:
            writer.close()
        env.close()
