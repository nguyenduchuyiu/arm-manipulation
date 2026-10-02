"""Closed-loop MuJoCo rollout of the joint 25 Hz TCOW/flow checkpoint."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from memory_occlusion.dataset.episode import context
from memory_occlusion.environment.env import MemoryOcclusionEnv
from memory_occlusion.experiments.tcow_joint_flow.visualization import mask_panel
from memory_occlusion.experiments.tcow_joint_flow.model import expand_depth_channel, restore_joint_model
from memory_occlusion.experiments.tcow_joint_flow.tcow import Seeker
from memory_occlusion.task import normalized


def physical_action(action, limits):
    fraction = np.r_[(action[:5] + 1) / 2, action[5]]
    return (limits[:, 0] + fraction * np.diff(limits, axis=1)[:, 0]).astype(np.float32)


@torch.inference_mode()
def infer(model, rgbs, depths, query_visible, proprio=None, noise=None):
    device = next(model.parameters()).device
    index = np.rint(np.linspace(0, len(rgbs) - 1, 30)).astype(int)
    rgb = torch.from_numpy(np.stack([rgbs[i] for i in index])).permute(3, 0, 1, 2).to(device)
    rgb = (rgb.float() / 255 - .45) / .225
    depth = torch.from_numpy(np.stack([depths[i] for i in index])).to(device)
    depth = ((depth.clamp(.4, 1.6) - 1.0) / .6)[None]
    rgbd = torch.cat((rgb, depth), dim=0)[None]
    query = torch.zeros((1, 1, 30, 240, 320), device=device)
    query[0, 0, 0] = torch.from_numpy(query_visible).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        model._latent = None
        logits, _ = model.tcow(rgbd, query)
        latent = model._latent
        if proprio is not None:
            state = torch.from_numpy(proprio[None]).to(device)
            estimate = model.flow.sample(latent, state, noise)
    masks = (logits[0, :, -1].float() > 0).cpu().numpy()
    chunk = estimate[0].float().cpu().numpy() if proprio is not None else None
    return chunk, masks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reference_episode", type=Path,
                   help="Only episode metadata defines scene seed, query and context plan")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-total-frames", type=int, default=1000,
                   help="25 Hz video frames from the first reveal frame, including shuffle")
    p.add_argument("--execute-chunk", type=int, default=10)
    args = p.parse_args()
    torch.set_num_threads(4)
    config = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    seeker_args = dict(config["seeker_args"], tracker_pretrained=False)
    tcow = expand_depth_channel(Seeker(logging.getLogger("tcow"), **seeker_args))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model = restore_joint_model(tcow, saved).cuda().eval()
    run_rollout(model, args, saved["step"])


@torch.inference_mode()
def run_rollout(model, args, checkpoint_step):
    model.eval()
    if not 1 <= args.execute_chunk <= 25:
        raise ValueError("execute-chunk must be between 1 and the 25-step policy horizon")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    meta = json.loads((args.reference_episode / "episode.json").read_text())
    if meta["fps"] != 25 or meta["resolution"] != [240, 320]:
        raise ValueError("expected 25 Hz 240x320 reference scene")
    context_frames = int(meta["decision_frames"]["t_occ"])
    action_budget = args.max_total_frames - context_frames
    if action_budget < 1:
        raise ValueError(f"max-total-frames must exceed {context_frames} context frames")
    env = MemoryOcclusionEnv(context_fps=25, resolution=320, max_episode_steps=action_budget)
    env.reset(seed=meta["seed"], options={"upright_targets": True, "layout": meta["layout"]})
    env.data.time = 0.0
    target = meta["target_object_id"]
    target_geom = env.model.geom(target + "_visual").id
    rgbs, depths = [], []
    query_visible = None
    def record(_phase, _valid):
        nonlocal query_visible
        rgb, depth = env._overview()
        rgbs.append(rgb[40:280].copy())
        depths.append(depth[40:280].copy())
        if query_visible is None:
            query_visible = (env.segmentation()[40:280, :, 0] == target_geom)
    context(env, meta["context_plan"], record)
    if len(rgbs) != meta["decision_frames"]["t_occ"] or not query_visible.any():
        raise RuntimeError("live context does not match reference metadata")
    Image.fromarray(query_visible.astype(np.uint8) * 255).save(args.output / "query_mask.png")
    env.query(target)
    env.phase = "execute"
    cover_start = {name: env.data.xpos[env.model.body(name).id].copy()
                   for name in env.cover_qadr}
    initial_target_z = float(env.target_positions()[target][2])
    limits = env.model.actuator_ctrlrange.copy()
    fixed_noise = model.flow.sample_noise(1, "cpu", torch.Generator().manual_seed(0)).to(
        next(model.parameters()).device)
    writer = iio.imopen(args.output / "rollout_masks_25hz.mp4", "w", plugin="pyav")
    writer.init_video_stream("libx264", fps=25, pixel_format="yuv420p")
    trace = []
    policy_chunks = []
    context_calls = 0
    policy_calls = 0
    max_lift = 0.0
    reason = "max_total_frames"
    try:
        # Process the full scripted context before the first policy action.
        for index in tqdm(range(len(rgbs)), desc="TCOW context", unit="frame",
                          mininterval=5, file=sys.stdout):
            prefix_rgb = rgbs[:index + 1]
            prefix_depth = depths[:index + 1]
            _, masks = infer(model, prefix_rgb, prefix_depth, query_visible)
            writer.write(mask_panel(rgbs[index], masks, index, index,
                                    query_visible if index == 0 else None), is_batch=False)
            context_calls += 1
        observation = env.observe()
        rgbs.append(observation["overview_rgb"][40:280].copy())
        depths.append(observation["overview_depth_m"][40:280].copy())
        print(json.dumps({"event": "context_ready", "frames": len(rgbs) - 1,
                          "tcow_context_calls": context_calls,
                          "query_pixels": int(query_visible.sum())}), flush=True)
        with tqdm(total=action_budget, desc="closed loop", unit="step",
                  mininterval=5, file=sys.stdout) as progress:
            while len(trace) < action_budget:
                source_frame = len(rgbs) - 1
                joints = normalized(env.data.qpos[env.robot_qpos_addresses], limits)
                chunk, masks = infer(model, rgbs, depths, query_visible, joints, fixed_noise)
                policy_chunks.append({"source_frame": source_frame,
                                      "proprio": joints.tolist(),
                                      "action_25_raw": chunk.tolist()})
                policy_calls += 1
                done = False
                for offset in range(min(args.execute_chunk, action_budget - len(trace))):
                    frame_index = len(rgbs) - 1
                    joints = normalized(env.data.qpos[env.robot_qpos_addresses], limits)
                    action = chunk[offset].copy()
                    action[:5] = np.clip(action[:5], -1, 1)
                    action[5] = float(action[5] >= .5)
                    writer.write(mask_panel(rgbs[-1], masks, frame_index, source_frame),
                                 is_batch=False)
                    observation, _, terminated, truncated, _ = env.step(
                        physical_action(action, limits))
                    max_lift = max(max_lift, float(env.target_positions()[target][2] - initial_target_z))
                    movement = {name: float(np.linalg.norm(env.data.xpos[env.model.body(name).id] - start))
                                for name, start in cover_start.items()}
                    if env.selected_cover is None:
                        selected = max(movement, key=movement.get)
                        if movement[selected] > .04:
                            env.selected_cover = selected
                    trace.append({"frame": frame_index, "source_frame": source_frame,
                                  "chunk_offset": offset, "action": action.tolist(),
                                  "proprio": joints.tolist(), "selected_cover": env.selected_cover,
                                  "cover_displacement_m": movement,
                                  "target_lift_m": max_lift, "success": bool(env.success),
                                  "failure_reason": env.failure_reason})
                    rgbs.append(observation["overview_rgb"][40:280].copy())
                    depths.append(observation["overview_depth_m"][40:280].copy())
                    progress.update(1)
                    if terminated or truncated:
                        reason = env.failure_reason or "success"
                        done = True
                        break
                if done:
                    break
    finally:
        writer.close()
        env.close()
    with (args.output / "trace.jsonl").open("w") as stream:
        for row in trace:
            stream.write(json.dumps(row) + "\n")
    with (args.output / "policy_chunks.jsonl").open("w") as stream:
        for row in policy_chunks:
            stream.write(json.dumps(row) + "\n")
    result = {"reference_episode": str(args.reference_episode), "checkpoint_step": checkpoint_step,
              "target": target, "seed": meta["seed"], "layout": meta["layout"],
              "query_pixels": int(query_visible.sum()), "context_frames": context_calls,
              "action_steps": len(trace), "fps": 25, "video_frames": context_calls + len(trace),
              "max_total_frames": args.max_total_frames, "action_budget": action_budget,
              "execute_chunk": args.execute_chunk, "policy_calls": policy_calls,
              "selected_cover": env.selected_cover, "correct_cover": meta["correct_cover_body"],
              "cover_choice_correct": env.selected_cover == meta["correct_cover_body"],
              "cover_removed": bool(env._cover_in_drop_zone() and
                                    env.selected_cover == meta["correct_cover_body"]),
              "target_grasped": bool(env.target_grasped),
              "target_lift_m": max_lift,
              "success": bool(env.success and env.selected_cover == meta["correct_cover_body"]),
              "failure_reason": env.failure_reason, "stop_reason": reason,
              "video": str(args.output / "rollout_masks_25hz.mp4"),
              "note": "Only the first-frame visible GT query mask is supplied to TCOW; all later RGB-D and proprio are live sim observations."}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"event": "complete", **result}), flush=True)
    return result


if __name__ == "__main__":
    main()
