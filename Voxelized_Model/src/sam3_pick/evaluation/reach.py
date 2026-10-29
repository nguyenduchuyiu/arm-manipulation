"""Fixed held-out chunk metrics and H25/K15 physical reach rollouts."""

import mujoco
import numpy as np
import torch
from tqdm import tqdm
from sam3_pick.data.reach_clouds import DATA
from sam3_pick.simulation.pose import ARM_JOINTS, PoseIK, rotation_matrix

GRID = DATA.parent


@torch.no_grad()
def chunks(policy, clouds, arrays, ids, seed=79, sample=False):
    total = count = 0
    errors = []
    rotations = []
    jaws = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        for begin in tqdm(
            range(0, len(ids), 256),
            desc="Fixed reach chunks",
            mininterval=5,
            leave=False,
        ):
            rows = ids[begin : begin + 256]
            cloud = torch.tensor(np.array(clouds[arrays["episode"][rows]]), device="cuda")
            state = torch.tensor(np.array(arrays["joints"][rows]), device="cuda")
            action = torch.tensor(np.array(arrays["actions"][rows]), device="cuda")
            mask = torch.tensor(np.array(arrays["mask"][rows]), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = policy.loss(cloud, state, action, mask)
                if sample:
                    noise = policy.flow.sample_noise(len(rows), "cuda")
                    prediction = policy.sample(cloud, state, noise).float()
            weight = int(mask.sum())
            total += float(loss) * weight
            count += weight
            if sample:
                p = prediction.cpu().numpy()
                a = action.cpu().numpy()
                m = mask.cpu().numpy()
                errors.extend(np.linalg.norm(p[..., :3] - a[..., :3], axis=-1)[m].tolist())
                r = rotation_matrix(p[..., 3:9]) @ rotation_matrix(a[..., 3:9]).swapaxes(-1, -2)
                angles = np.arccos(np.clip((np.trace(r, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
                rotations.extend(np.rad2deg(angles[m]).tolist())
                jaws.extend(np.abs(p[..., 9] - a[..., 9])[m].tolist())
    result = dict(chunks=len(ids), valid_actions=count, FM_loss=total / count, seed=seed)
    if sample:
        e = np.asarray(errors) * 1000
        result.update(
            EE_delta_mean_mm=float(e.mean()),
            EE_delta_median_mm=float(np.median(e)),
            EE_delta_p90_mm=float(np.quantile(e, 0.9)),
            rotation_mean_deg=float(np.mean(rotations)),
            gripper_MAE_mm=float(np.mean(jaws) * 1000),
        )
    return result


def physics():
    model = mujoco.MjModel.from_xml_path(str(GRID / "scene.xml"))
    joint_ids = [model.joint(n).id for n in ARM_JOINTS]
    joint_ids.append(model.joint("right_jaw_slide_joint").id)
    qadr = model.jnt_qposadr[joint_ids]
    dofs = model.jnt_dofadr[joint_ids[:5]]
    return (model, qadr, dofs, model.site("grasp_site").id)


@torch.no_grad()
def rollouts(policy, clouds, trajectory_ids, rows, seed=0, oracle=False, paired_noise=False):
    robot = physics()
    model, qadr, _, site = robot
    ik = PoseIK(robot)
    cases = []
    for index in trajectory_ids:
        row = rows[int(index)]
        source = np.load(GRID / row["source_trajectory_file"])
        state = mujoco.MjData(model)
        state.qpos[:] = source["state_full_qpos"][0]
        state.qvel[:] = source["state_full_qvel"][0]
        state.ctrl[:] = source["initial_ctrl"]
        mujoco.mj_forward(model, state)
        case = dict(
            data=state,
            held=state.ctrl.copy(),
            target=np.asarray(row["target_world_m"]),
            source={
                k: source[k].copy()
                for k in ("ee_position_world_m", "action_absolute", "state_full_qpos")
            },
            ee=[state.site_xpos[site].copy()],
            qpos=[state.qpos.copy()],
            commands=[],
            jaw=[],
            speed=[],
            collisions=False,
            max_penetration=0.0,
            clipped_frames=0,
        )
        source.close()
        cases.append(case)
    noise_rng = torch.Generator(device="cuda").manual_seed(seed)
    for frame in tqdm(
        range(300),
        desc=f"Reach {len(cases)} goals noise{seed}",
        mininterval=5,
        leave=False,
    ):
        if frame % 15 == 0 and (not oracle):
            inputs = torch.tensor(
                np.array(clouds[trajectory_ids]), dtype=torch.float32, device="cuda"
            )
            joints = torch.tensor(
                np.array([c["data"].qpos[qadr] for c in cases]),
                dtype=torch.float32,
                device="cuda",
            )
            noise = policy.flow.sample_noise(1 if paired_noise else len(cases), "cuda", noise_rng)
            if paired_noise:
                noise = noise.expand(len(cases), -1, -1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                plans = policy.sample(inputs, joints, noise, steps=10).float().cpu().numpy()
            for case in cases:
                case["anchor"] = case["data"].site_xpos[site].copy()
                case["rotation"] = case["data"].site_xmat[site].reshape(3, 3).copy()
        for i, case in enumerate(cases):
            state = case["data"]
            if oracle:
                src = case["source"]
                t = min(2 * (frame + 1), len(src["state_full_qpos"]) - 1)
                old = mujoco.MjData(model)
                old.qpos[:] = src["state_full_qpos"][t]
                mujoco.mj_kinematics(model, old)
                target = src["ee_position_world_m"][t]
                rotation = old.site_xmat[site].reshape(3, 3)
                jaw = src["action_absolute"][min(2 * frame, len(src["action_absolute"]) - 1), 5]
            else:
                action = plans[i, frame % 15]
                target = case["anchor"] + action[:3]
                rotation = rotation_matrix(action[3:9]) @ case["rotation"]
                jaw = action[9]
            raw = ik.command(case, target, jaw, rotation)
            command = np.clip(raw, *model.actuator_ctrlrange.T)
            case["clipped_frames"] += int(np.any(raw != command))
            for servo in ((case["held"] + command) * 0.5, command):
                state.ctrl[:] = servo
                for _ in range(10):
                    mujoco.mj_step(model, state)
                    penetration = max((max(0.0, -c.dist) for c in state.contact), default=0.0)
                    case["max_penetration"] = max(case["max_penetration"], penetration)
                    case["collisions"] |= penetration > 0.0002
            mujoco.mj_forward(model, state)
            assert np.isfinite(state.qpos).all()
            position = state.site_xpos[site].copy()
            case["speed"].append(float(np.linalg.norm(position - case["ee"][-1]) * 25))
            case["ee"].append(position)
            case["qpos"].append(state.qpos.copy())
            case["jaw"].append(float(state.qpos[qadr[-1]]))
            case["commands"].append(command.copy())
            case["held"] = command.copy()
    records = []
    for index, case in zip(trajectory_ids, cases):
        errors = np.linalg.norm(np.asarray(case["ee"]) - case["target"], axis=-1)
        stable = (np.abs(np.asarray(case["jaw"])[-25:] + 0.0255) < 0.001).all()
        stable &= (np.asarray(case["speed"])[-25:] < 0.01).all() and (not case["collisions"])
        record = dict(
            trajectory_index=int(index),
            voxel_id=rows[int(index)]["voxel_id"],
            final_error_mm=float(errors[-1] * 1000),
            minimum_error_mm=float(errors.min() * 1000),
            collision=bool(case["collisions"]),
            max_penetration_mm=case["max_penetration"] * 1000,
            clipped_frames=case["clipped_frames"],
            ik_mean_residual_mm=float(np.mean(case["ik_errors"]) * 1000),
        )
        for mm in (1, 2, 5, 10, 20):
            record[f"success_{mm}mm"] = bool(stable and (errors[-25:] < mm / 1000).all())
        records.append(record)
    final = [r["final_error_mm"] for r in records]
    summary = dict(
        cases=len(records),
        final_mean_mm=float(np.mean(final)),
        final_median_mm=float(np.median(final)),
        final_p90_mm=float(np.quantile(final, 0.9)),
        collision_rate=float(np.mean([r["collision"] for r in records])),
        clipped_frame_fraction=float(np.mean([r["clipped_frames"] / 300 for r in records])),
        seed=seed,
        success_definition="last25 frames: EE error below threshold, measured jaw within1mm closed, EE finite-difference speed below10mm/s, no penetration>0.2mm at any physics substep",
    )
    for mm in (1, 2, 5, 10, 20):
        summary[f"success_{mm}mm"] = float(np.mean([r[f"success_{mm}mm"] for r in records]))
    arrays = dict(
        ee=np.asarray([c["ee"] for c in cases], np.float32),
        qpos=np.asarray([c["qpos"] for c in cases], np.float32),
        action=np.asarray([c["commands"] for c in cases], np.float32),
        target=np.asarray([c["target"] for c in cases]),
        trajectory_ids=trajectory_ids,
    )
    return (dict(summary=summary, records=records), arrays)
