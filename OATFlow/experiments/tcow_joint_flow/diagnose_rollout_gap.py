"""Compare expert-state action chunks at rollout start with the closed-loop trace."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from OATFlow.experiments.tcow_joint_flow.data import clip, load_episode
from OATFlow.experiments.tcow_joint_flow.model import expand_depth_channel, restore_joint_model
from OATFlow.experiments.tcow_joint_flow.tcow import Seeker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config-checkpoint", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    episode = load_episode(args.episode)
    trace = [json.loads(line) for line in args.trace.read_text().splitlines()]
    start = trace[0]["frame"]
    ends = [start + 25 * i for i in range(5)]
    if any(t not in episode["valid_ends"] for t in ends):
        raise ValueError(f"one of the queried chunks has invalid expert actions: {ends}")
    config = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    seeker_args = dict(config["seeker_args"])
    seeker_args["tracker_pretrained"] = False
    tcow = expand_depth_channel(Seeker(logging.getLogger("tcow"), **seeker_args))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model = restore_joint_model(tcow, saved).cuda().eval()
    samples = [clip(episode, end, "cuda") for end in ends]
    rgbd, query, truth, proprio, action = (torch.cat(items, dim=0)
                                            for items in zip(*samples))
    fixed_noise = model.flow.sample_noise(1, "cpu", torch.Generator().manual_seed(0))
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model._latent = None
        masks, _ = model.tcow(rgbd, query)
        latent = model._latent
        estimate = model.flow.sample(latent, proprio, fixed_noise.cuda().repeat(len(ends), 1, 1))
    raw = estimate.float().cpu().numpy()
    predicted = raw.copy()
    predicted[:, :, :5] = np.clip(predicted[:, :, :5], -1, 1)
    predicted[:, :, 5] = (predicted[:, :, 5] >= .5).astype(np.float32)
    expert = action.cpu().numpy()
    hold = np.broadcast_to(proprio.cpu().numpy()[:, None], expert.shape)
    rows = []
    for index, end in enumerate(ends):
        simulated = np.array([row["action"] for row in trace
                              if end <= row["frame"] < end + 25])
        diff = np.abs(predicted[index] - expert[index])
        raw_diff = np.abs(raw[index] - expert[index])
        rows.append({"frame": end, "expert_state_action_mae": float(diff.mean()),
                     "expert_state_raw_mae": float(raw_diff.mean()),
                     "expert_state_arm_mae": float(diff[:, :5].mean()),
                     "expert_state_gripper_mae": float(diff[:, 5].mean()),
                     "hold_mae": float(np.abs(hold[index] - expert[index]).mean()),
                     "first_expert_action": expert[index, 0].tolist(),
                     "first_expert_state_prediction": predicted[index, 0].tolist(),
                     "first_closed_loop_action": simulated[0].tolist() if len(simulated) else None,
                     "closed_loop_vs_expert_mae": float(np.abs(simulated - expert[index, :len(simulated)]).mean())
                     if len(simulated) else None,
                     "expert_gripper_closed_frames": int(np.sum(expert[index, :, 5] < .5)),
                     "predicted_gripper_closed_frames": int(np.sum(predicted[index, :, 5] < .5))})
    output = {"episode": str(args.episode), "checkpoint_step": saved["step"],
              "protocol": "same fixed Gaussian source and 8 Euler steps as rollout; expert-state clips at five replanning frames",
              "rows": rows}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
