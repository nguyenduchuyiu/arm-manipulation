"""Evaluate learned FM with simulator-provided task IDs and absolute joint execution."""
import argparse
import json
from pathlib import Path
import sys
import time

import imageio.v3 as iio
import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from OATFlow.task import physical_joint_action
from OATFlow.environment.env import TARGETS, MemoryOcclusionEnv
from OATFlow.environment.control import step_joint_waypoint
from OATFlow.task import task_id_from_state
from OATFlow.prior.model import PriorPolicy
from OATFlow.prior.execution import AsyncPlanner, TemporalPlans
from OATFlow.environment.success import cover_deposited
from OATFlow.task import normalized


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS,
                        help="override target in the same scene; derive its pair/cover task from physical state")
    parser.add_argument("--execute-chunk", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=1800)
    parser.add_argument("--temporal-ensemble-coeff", type=float,
                        help="ACT exponential ensembling; use 0.01 with --execute-chunk 1")
    parser.add_argument("--servo-hz", type=int, choices=(25, 50), default=25,
                        help="50: q_k at0ms, midpoint at20ms, q_(k+1) at40ms")
    parser.add_argument("--execution-mode", choices=("sync", "async"), default="sync",
                        help="async: background planning, ensemble/commit five waypoints, wall-clock servo50")
    args = parser.parse_args()
    asynchronous = args.execution_mode == "async"
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.execute_chunk <= 25 or args.max_frames < 1:
        raise ValueError("K must be 1–25; positive frame budget")
    if args.temporal_ensemble_coeff is not None and not np.isfinite(args.temporal_ensemble_coeff):
        raise ValueError("temporal ensembling requires a finite coefficient")
    if asynchronous and (args.execute_chunk != 5 or args.servo_hz != 50 or args.temporal_ensemble_coeff is None):
        raise ValueError("async requires --execute-chunk 5 --servo-hz 50 --temporal-ensemble-coeff 0.01")
    if not asynchronous and args.temporal_ensemble_coeff is not None and args.execute_chunk != 1:
        raise ValueError("synchronous ACT temporal ensembling requires --execute-chunk 1")
    if not asynchronous and args.servo_hz == 50 and (args.execute_chunk != 1 or args.temporal_ensemble_coeff is None):
        raise ValueError("50Hz ACT pipeline requires execute1 and temporal ensembling")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if saved["architecture"] != "oracle_task_target_fm":
        raise ValueError("oracle checkpoint required")
    representation = saved["config"]["action_representation"]
    if representation != "absolute_joint":
        raise ValueError("absolute joint checkpoint required")
    model = PriorPolicy().cuda().eval()
    model.load_state_dict(saved["model"], strict=True)
    ensembler = None
    use_ensemble = args.temporal_ensemble_coeff is not None
    if use_ensemble and not asynchronous:
        from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
        ensembler = ACTTemporalEnsembler(args.temporal_ensemble_coeff, model.flow.horizon)
    meta = json.loads((args.episode / "episode.json").read_text())
    env = MemoryOcclusionEnv(resolution=320)
    with np.load(args.episode / "sim_state.npz") as initial:
        env.data.qpos[:] = initial["initial_qpos"]
        env.data.qvel[:] = initial["initial_qvel"]
        env.data.ctrl[:] = initial["initial_ctrl"]
    env.assignment = meta["object_to_occluder"]
    mujoco.mj_forward(env.model, env.data)
    if task_id_from_state(env, meta["task"]["objects"]) != meta["task_id"]:
        raise ValueError("reference oracle ID differs from simulator state")
    initial_positions = env.target_positions()
    reference_target = meta["target_object_id"]
    if meta["target_id"] != TARGETS.index(reference_target):
        raise ValueError("reference target identity differs from embedding ID")
    target = args.target or reference_target
    cover = env.assignment[target]
    pair = [name for name in TARGETS if env.assignment[name] == cover]
    task_id = task_id_from_state(env, pair)
    if args.target is None and (task_id != meta["task_id"] or cover != meta["selected_cover"]):
        raise ValueError("reference query differs from physical task and cover")
    target_id = TARGETS.index(target)
    jaws = {env.model.geom(f"link_6_{side}_jaw_collision_0").id for side in ("left", "right")}
    body = env.model.body(target).id
    args.output.mkdir(parents=True)
    config = dict(checkpoint=str(args.checkpoint), step=saved["step"], epoch=saved["epoch"],
                  reference=str(args.episode), seed=meta["seed"], task_id=task_id,
                  target=target, target_id=target_id, reference_target=reference_target,
                  reference_task_id=meta["task_id"], selected_cover=cover, task_pair=pair,
                  target_override=args.target is not None,
                  horizon=25, execute_chunk=args.execute_chunk, flow_steps=10,
                  noise_seed=0, max_frames=args.max_frames, oracle_task_ids=True, control="learned_policy",
                  action_representation=representation)
    config.update(temporal_ensemble=use_ensemble, execution_mode=args.execution_mode,
                  policy_hz=25, servo_hz=args.servo_hz, timebase="wall-paced simulation" if asynchronous else "simulation",
                  interpolation="linear midpoint between successive absolute waypoints" if args.servo_hz == 50 else None,
                  command_timeline="q_k@0ms, midpoint@20ms, q_(k+1)@40ms" if args.servo_hz == 50 else "new setpoint every40ms",
                  trace_command_semantics="command:25Hz target waypoint; servo_command:timestamped actual commands",
                  cpu_threads=2,
                  temporal_ensemble_coeff=args.temporal_ensemble_coeff,
                  temporal_ensemble_source="timestamped TemporalPlans" if asynchronous else "lerobot.ACTTemporalEnsembler" if ensembler else None,
                  ensemble_weighting="exp(-coeff*i), i=0 oldest prediction" if use_ensemble else None)
    if asynchronous:
        config.update(inference_cadence="continuous; render latest copied simulator state after each completion",
                      plan_alignment="action[j] from source_step ends at (source_step+j+1)/25 seconds",
                      commitment="five waypoints frozen at each block boundary",
                      underrun="hold last commanded waypoint; record ensemble_count=0",
                      startup="prime first H25 plan before starting servo clock",
                      plan_latency_includes="camera rendering, input transfer and FM inference",
                      video="render recorded simulator states after execution; no camera/video work on servo thread",
                      late_commands="shift wall schedule forward; no catch-up bursts")
    config["training"] = saved["config"]
    config["objective"] = "cover_drop_then_target_lift"
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "README.md").write_text(
        f"# {args.output.name}\n\nLearned FM policy rollout with simulator-provided task IDs, checkpoint {args.checkpoint}, step {saved['step']}.\n"
        f"Reference {args.episode}; seed {meta['seed']}, task {task_id}, target {target}.\n"
        f"H25/K{args.execute_chunk}; Euler10; noise0; 25Hz; budget {args.max_frames} frames.\n"
        f"Actions: {representation}; joint actions decode to actuator setpoints directly.\n"
        f"ACT temporal ensembling: {use_ensemble}; coefficient {args.temporal_ensemble_coeff}; "
        "overlapping chunks aligned at the same control timestep, all six dimensions averaged before clipping.\n"
        f"Policy25Hz; servo{args.servo_hz}Hz. At50Hz send q_k@0ms, midpoint@20ms, q_(k+1)@40ms. "
        f"Execution mode: {args.execution_mode}; clock: {config['timebase']}. Inference wall latency is recorded separately.\n"
        "Success: cover released/settled in broad green zone, correct target lifted >4cm with both jaws for 10 frames.\n"
        f"Training: data {saved['config']['data']}; epochs {saved['config']['epochs']}; batch {saved['config']['batch_size']}; "
        f"warmup/cosine LR {saved['config']['lr']} → {saved['config']['min_lr']}. "
        f"Vision source {saved['config']['vision_weights']}; FM source {saved['config']['flow_weights']}. "
        "Frozen shared ImageNet ViT-B/16; train visual/proprio adapters, task/target/view embeddings, context decoder and FM.\n"
        "Full training/test configuration: config.json. Status: running.\n")
    writer = iio.imopen(args.output / "rollout.mp4", "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=25, pixel_format="yuv420p")
    generator = torch.Generator(device="cuda").manual_seed(0)
    chunk, held = None, 0
    result = dict(success=False, reason="frame_budget", frames=0, max_target_lift_m=0.,
                  target_grasped=False, target_lifted=False,
                  cover_removed=False, first_cover_removed_frame=None, max_cover_lift_m=0.)
    result.update(temporal_ensemble=use_ensemble, execute_chunk=args.execute_chunk,
                  execution_mode=args.execution_mode,
                  policy_hz=25, servo_hz=args.servo_hz,
                  inference_calls=0, inference_wall_s=0.)
    result.update(wrong_object=None, max_object_lift_m={name: 0. for name in TARGETS})
    initial_cover_z = float(env.data.xpos[env.model.body(cover).id, 2])
    trace = {key: [] for key in ("joint_position", "command", "target_xyz", "cover_xyz", "target_jaw_contacts", "object_xyz")}
    trace.update(raw_first_action=[], ensembled_action=[], ensemble_count=[])
    trace.update(waypoint_timestamp_s=[], servo_timestamp_s=[], servo_command=[])
    planner = None
    if asynchronous:
        plans = TemporalPlans(args.temporal_ensemble_coeff, model.flow.horizon)
        planning_data = mujoco.MjData(env.model)
        planning_renderer = None
        video_states = []
        trace.update(plan_source_step=[], plan_received_step=[], plan_actions=[], plan_latency_s=[],
                     plan_completion_wall_s=[], committed_block_start=[], servo_wall_timestamp_s=[])
        result.update(underrun_steps=0, expired_plans=0, skipped_prediction_actions=0)

    @torch.inference_mode()
    def infer(overview, wrist, proprio):
        views = [torch.from_numpy(view).permute(2, 0, 1)[None].cuda() for view in (overview, wrist)]
        noise = model.flow.sample_noise(1, "cuda", generator)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actions = model.sample(*views, torch.from_numpy(proprio[None]).cuda(),
                                   torch.tensor([task_id], device="cuda"),
                                   torch.tensor([target_id], device="cuda"), noise).float()
        if not torch.isfinite(actions).all():
            raise ValueError("nonfinite policy prediction")
        return actions

    def predict(snapshot):
        nonlocal planning_renderer
        if planning_renderer is None:
            planning_renderer = mujoco.Renderer(env.model, height=320, width=320)
        restore_state(planning_data, snapshot)
        planning_renderer.update_scene(planning_data, camera="overview")
        overview = planning_renderer.render().copy()
        planning_renderer.update_scene(planning_data, camera="wrist")
        wrist = planning_renderer.render().copy()
        proprio = normalized(planning_data.qpos[env.robot_qpos_addresses], env.model.actuator_ctrlrange)
        return infer(overview, wrist, proprio)[0].cpu().numpy()

    def close_planning_renderer():
        if planning_renderer is not None:
            planning_renderer.close()

    def capture_state():
        return env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), float(env.data.time)

    def restore_state(data, snapshot):
        qpos, qvel, ctrl, sim_time = snapshot
        data.qpos[:], data.qvel[:], data.ctrl[:], data.time = qpos, qvel, ctrl, sim_time
        mujoco.mj_forward(env.model, data)

    def accept_plans(completed, frame):
        for source, actions, latency, finished in completed:
            plans.add(source, actions, frame)
            trace["plan_source_step"].append(source)
            trace["plan_received_step"].append(frame)
            trace["plan_actions"].append(actions)
            trace["plan_latency_s"].append(latency)
            trace["plan_completion_wall_s"].append(finished - rollout_started)
            result["inference_calls"] += 1
            result["inference_wall_s"] += latency
            result["skipped_prediction_actions"] += min(frame - source, model.flow.horizon)
            result["expired_plans"] += frame - source >= model.flow.horizon

    if args.servo_hz == 50:
        trace["servo_timestamp_s"].append(float(env.data.time))
        trace["servo_command"].append(env.data.ctrl.copy())
    try:
        if asynchronous:
            first_snapshot = capture_state()
            planner = AsyncPlanner(predict, cleanup=close_planning_renderer)
            planner.publish(0, first_snapshot)
            primed = planner.take(wait=True)
            rollout_started = time.perf_counter()
            servo_started = rollout_started
            accept_plans(primed, 0)
            trace["servo_wall_timestamp_s"].append(0.)
        else:
            rollout_started = time.perf_counter()
        for frame in tqdm(range(args.max_frames), desc="FM policy rollout", unit="frame", mininterval=5, file=sys.stdout):
            if asynchronous:
                snapshot = capture_state()
                video_states.append(snapshot)
                planner.publish(frame, snapshot)
                accept_plans(planner.take(), frame)
                if frame % 5 == 0:
                    block, raw_block, block_counts = plans.block(frame, 5, normalized(env.data.ctrl, env.model.actuator_ctrlrange))
                offset = frame % 5
                action = block[offset]
                raw_action = raw_block[offset]
                ensemble_count = int(block_counts[offset])
                result["underrun_steps"] += ensemble_count == 0
                trace["committed_block_start"].append(frame - offset)
            else:
                overview, wrist = env._overview(), env.wrist_image()
                writer.write(np.concatenate((overview, wrist), axis=1), is_batch=False)
                proprio = normalized(env.data.qpos[env.robot_qpos_addresses], env.model.actuator_ctrlrange)
            if not asynchronous and frame % args.execute_chunk == 0:
                inference_started = time.perf_counter()
                actions = infer(overview, wrist, proprio)
                chunk = actions[0].cpu().numpy()
                if ensembler is not None:
                    ensembled_action = ensembler.update(actions)[0].cpu().numpy()
                result["inference_calls"] += 1
                result["inference_wall_s"] += time.perf_counter() - inference_started
            if not asynchronous:
                action = ensembled_action if ensembler is not None else chunk[frame % args.execute_chunk]
                raw_action = chunk[0]
                ensemble_count = min(frame + 1, 25) if ensembler is not None else 1
            trace["raw_first_action"].append(raw_action.copy())
            trace["ensembled_action"].append(action.copy())
            trace["ensemble_count"].append(ensemble_count)
            command = physical_joint_action(action, env.model.actuator_ctrlrange)
            command = np.clip(command, *env.model.actuator_ctrlrange.T).astype(np.float32)
            if args.servo_hz == 50:
                wall_times = [] if asynchronous else None
                times, commands = step_joint_waypoint(env.model, env.data, command,
                                                     start_time=servo_started if asynchronous else None,
                                                     wall_times=wall_times)
                if asynchronous:
                    trace["servo_wall_timestamp_s"].extend(wall - rollout_started for wall in wall_times)
                    servo_started = wall_times[-1]
                trace["servo_timestamp_s"].extend(times)
                trace["servo_command"].extend(commands)
                trace["waypoint_timestamp_s"].append(float(times[-1]))
            else:
                trace["servo_timestamp_s"].append(float(env.data.time))
                trace["servo_command"].append(command.copy())
                trace["waypoint_timestamp_s"].append(float(env.data.time))
                env.data.ctrl[:] = command
                mujoco.mj_step(env.model, env.data, 20)
            result["frames"] = frame + 1
            positions = env.target_positions()
            for name in TARGETS:
                result["max_object_lift_m"][name] = max(result["max_object_lift_m"][name],
                    float(positions[name][2] - initial_positions[name][2]))
            gain = float(positions[target][2] - initial_positions[target][2])
            result["max_target_lift_m"] = max(result["max_target_lift_m"], gain)
            touching = {jaw for contact in env.data.contact for jaw in jaws
                        if ((contact.geom1 == jaw and env.model.geom_bodyid[contact.geom2] == body)
                            or (contact.geom2 == jaw and env.model.geom_bodyid[contact.geom1] == body))}
            cover_position = env.data.xpos[env.model.body(cover).id]
            cover_removed = cover_deposited(env.model, env.data, cover)
            result["cover_removed"] = bool(cover_removed)
            result["max_cover_lift_m"] = max(result["max_cover_lift_m"], float(cover_position[2] - initial_cover_z))
            if cover_removed and result["first_cover_removed_frame"] is None:
                result["first_cover_removed_frame"] = frame + 1
            trace["joint_position"].append(env.data.qpos[env.robot_qpos_addresses].copy())
            trace["command"].append(command.copy())
            trace["target_xyz"].append(positions[target].copy())
            trace["cover_xyz"].append(cover_position.copy())
            trace["target_jaw_contacts"].append(len(touching))
            trace["object_xyz"].append(np.stack([positions[name] for name in TARGETS]))
            wrong_objects = [name for name in TARGETS if name != target
                             and positions[name][2] > initial_positions[name][2] + .04]
            if wrong_objects:
                result["wrong_object"] = wrong_objects[0]
                result["reason"] = "wrong_object_lifted"
                break
            held = held + 1 if gain > .04 and touching == jaws and cover_removed else 0
            result["target_grasped"] |= held >= 10
            if (frame + 1) % 100 == 0:
                live = dict(status="running", sim_time_s=float(env.data.time),
                            servo_commands=len(trace["servo_command"]), **result)
                (args.output / "live.json").write_text(json.dumps(live, indent=2) + "\n")
                print(json.dumps({"event": "rollout_progress", **live}), flush=True)
            if held >= 10:
                result.update(success=True, target_lifted=True, reason="target_lifted")
                break
        result["servo_commands"] = len(trace["servo_command"])
        result["sim_duration_s"] = float(env.data.time)
        result["execution_wall_s"] = time.perf_counter() - rollout_started
        if asynchronous:
            planner.close()
            accept_plans(planner.take(), result["frames"])
            planner = None
            wall = np.asarray(trace["servo_wall_timestamp_s"])
            result.update(actual_servo_hz=(len(wall) - 1) / wall[-1],
                          actual_waypoint_hz=result["frames"] / wall[-1])
            print(json.dumps({"event": "render_recorded_rollout", **result}), flush=True)
            for snapshot in tqdm(video_states, desc="Render recorded rollout", unit="frame", mininterval=5, file=sys.stdout):
                restore_state(env.data, snapshot)
                writer.write(np.concatenate((env._overview(), env.wrist_image()), axis=1), is_batch=False)
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(args.output / "trace.npz", **{key: np.asarray(value) for key, value in trace.items()})
        path = args.output / "README.md"
        path.write_text(path.read_text().replace("Status: running", "Status: complete") + f"\nResults: {json.dumps(result)}\n")
        print(json.dumps(result), flush=True)
    finally:
        try:
            if planner is not None:
                planner.close()
        finally:
            writer.close()
            env.close()


if __name__ == "__main__":
    main()
