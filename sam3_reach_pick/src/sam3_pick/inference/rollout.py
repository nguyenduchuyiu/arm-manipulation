"""E133 closed-loop controller; observation supplies fused target XYZ only."""

import time
import mujoco
import numpy as np
import torch
from tqdm import tqdm
from sam3_pick.simulation.scene import Scene, NAMES
from sam3_pick.simulation.contacts import jaw_contacts
from sam3_pick.simulation.control import step as servo_step
from sam3_pick.simulation.pose import PoseIK, rotation_matrix
from sam3_pick.data.pick_labels import write


@torch.inference_mode()
def rollout(rows, policy, physics, observe, out, data_root, frame_budget=300, progress_path=None):
    scenes = []
    started = time.perf_counter()
    for row in rows:
        scene = Scene(physics)
        with np.load(data_root / row["path"] / "sim_state.npz") as z:
            for field in ("qpos", "qvel", "ctrl"):
                getattr(scene.data, field)[:] = z["initial_" + field]
        mujoco.mj_forward(physics, scene.data)
        state = scene.data
        scenes.append(
            dict(
                data=state,
                scene=scene,
                row=row,
                held=state.ctrl.copy(),
                rng=np.random.default_rng(
                    np.random.SeedSequence(
                        [131, row["layout_id"], row["permutation"], row["target_id"]]
                    )
                ),
                ik=PoseIK((physics, scene.qadr, scene.dofs[:5], scene.site)),
                initial_qpos=state.qpos.copy(),
                initial_qvel=state.qvel.copy(),
                initial_ctrl=state.ctrl.copy(),
                initial_objects=np.stack([state.xpos[physics.body(n).id] for n in NAMES[:4]]),
                qpos=[],
                commands=[],
                objects=[],
                contacts=[],
                clouds=[],
                cloud_replans=[],
                mask_history=[],
                observation_records=[],
                success=False,
                done=False,
                reason="timeout",
                missing_detail=None,
                streak=0,
                longest=0,
            )
        )
    generator = torch.Generator(device="cuda").manual_seed(0)
    for frame in tqdm(
        range(frame_budget),
        desc="Fused cloud policy H25/K15",
        leave=False,
        mininterval=10,
    ):
        active = [s for s in scenes if not s["done"]]
        if not active:
            break
        if frame % 15 == 0:
            usable = []
            clouds = []
            joints = []
            for s in tqdm(
                active,
                desc=f"Two-view observations f{frame}",
                leave=False,
                mininterval=10,
            ):
                cloud, good, stats, masks = observe(s["data"].qpos, s["row"]["target_id"], s["rng"])
                s["observation_records"].append(dict(frame=frame, valid=bool(good), **stats))
                s["mask_history"].append(np.packbits(masks, axis=-1, bitorder="little"))
                if not good:
                    empty = all((a.get("detections", 0) == 0 for a in stats["segmentation"]))
                    s.update(
                        done=True,
                        reason="missing_target_cloud",
                        missing_detail="no_text_detection"
                        if empty
                        else "insufficient_filtered_depth",
                    )
                    continue
                usable.append(s)
                clouds.append(cloud)
                joints.append(s["data"].qpos[s["scene"].qadr])
                s["cloud_replans"].append(frame)
                s["clouds"].append(cloud)
            if usable:
                cloud = torch.tensor(np.stack(clouds), device="cuda")
                joint = torch.tensor(np.stack(joints), device="cuda", dtype=torch.float32)
                noise = policy.flow.sample_noise(1, "cuda", generator).expand(len(usable), -1, -1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    plans = policy.sample(cloud, joint, noise, steps=10).float().cpu().numpy()
                for s, plan in zip(usable, plans):
                    assert np.isfinite(plan).all()
                    s["plan"] = plan
                    s["anchor"] = s["data"].site_xpos[s["scene"].site].copy()
                    s["rotation"] = s["data"].site_xmat[s["scene"].site].reshape(3, 3).copy()
            if progress_path is not None:
                write(
                    progress_path,
                    dict(
                        status="replanning",
                        physical_frame=frame,
                        batch_cases=len(rows),
                        active=len(usable),
                        finished=sum((s["done"] for s in scenes)),
                        elapsed_s=time.perf_counter() - started,
                    ),
                )
        for s in [s for s in active if not s["done"]]:
            state = s["data"]
            action = s["plan"][frame % 15]
            command = s["ik"].command(
                s,
                s["anchor"] + action[:3],
                action[9],
                rotation_matrix(action[3:9]) @ s["rotation"],
            )
            command = np.clip(command, *physics.actuator_ctrlrange.T)
            servo_step(physics, state, s["held"], command)
            s["held"] = command.copy()
            objects = np.stack([state.xpos[physics.body(n).id] for n in NAMES[:4]])
            contacts = np.array([len(jaw_contacts(physics, state, n)) for n in NAMES[:4]])
            gains = objects[:, 2] - s["initial_objects"][:, 2]
            target = s["row"]["target_id"]
            s["streak"] = s["streak"] + 1 if gains[target] > 0.04 and contacts[target] == 2 else 0
            s["longest"] = max(s["longest"], s["streak"])
            wrong = [i for i in range(4) if i != target and gains[i] > 0.04]
            if wrong:
                s.update(done=True, reason="wrong_object", wrong_object=NAMES[wrong[0]])
            elif s["streak"] >= 10:
                s.update(done=True, success=True, reason="success")
            assert np.isfinite(state.qpos).all()
            s["qpos"].append(state.qpos.copy())
            s["commands"].append(command.copy())
            s["objects"].append(objects)
            s["contacts"].append(contacts)
    results = []
    for s in scenes:
        row = s["row"]
        target = row["target_id"]
        gains = (
            np.asarray(s["objects"])[:, target, 2] - s["initial_objects"][target, 2]
            if s["objects"]
            else np.array([0.0])
        )
        record = dict(
            path=row["path"],
            group=row["evaluation_group"],
            layout=row["layout_id"],
            permutation=row["permutation"],
            target=NAMES[target],
            slot=row["target_slot"],
            success=s["success"],
            reason=s["reason"],
            missing_detail=s["missing_detail"],
            frames=len(s["commands"]),
            bilateral_lift_streak=s["longest"],
            wrong_object=s.get("wrong_object"),
            max_target_lift_mm=float(gains.max() * 1000),
            max_ik_error_mm=float(max(s.get("ik_errors", [0.0])) * 1000),
            observation_count=len(s["observation_records"]),
            single_view_observations=sum(
                (sum(r["usable_views"]) == 1 for r in s["observation_records"])
            ),
            dual_view_observations=sum(
                (sum(r["usable_views"]) == 2 for r in s["observation_records"])
            ),
        )
        for key in ("pair", "edge", "target_xy", "cell", "band_index"):
            if key in row:
                record[key] = row[key]
        name = row["path"].replace("/", "_")
        np.savez_compressed(
            out / (name + ".npz"),
            qpos=np.asarray(s["qpos"], np.float32).reshape(-1, physics.nq),
            commands=np.asarray(s["commands"], np.float32).reshape(-1, 6),
            objects=np.asarray(s["objects"], np.float32).reshape(-1, 4, 3),
            contacts=np.asarray(s["contacts"], np.uint8).reshape(-1, 4),
            initial_qpos=s["initial_qpos"],
            initial_qvel=s["initial_qvel"],
            initial_ctrl=s["initial_ctrl"],
            cloud_replan_frames=np.array(s["cloud_replans"]),
            input_cloud_base_m=np.asarray(s["clouds"], np.float32).reshape(-1, 128, 3),
            mask_replan_frames=np.array([r["frame"] for r in s["observation_records"]]),
            predicted_mask_packed=np.asarray(s["mask_history"], np.uint8).reshape(-1, 2, 240, 40),
            mask_width=320,
        )
        write(out / (name + "_observations.json"), s["observation_records"])
        results.append(record)
    return results
