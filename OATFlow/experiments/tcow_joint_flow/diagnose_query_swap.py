"""Check whether flow actions change when only the first-frame query changes."""
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
    p.add_argument("--episodes", type=Path, nargs=2, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    episodes = [load_episode(path) for path in args.episodes]
    metadata = [json.loads((path / "episode.json").read_text()) for path in args.episodes]
    if metadata[0]["seed"] != metadata[1]["seed"]:
        raise ValueError("query swap requires the same scene seed")
    frame = int(metadata[0]["decision_frames"]["t_occ"])
    if frame != int(metadata[1]["decision_frames"]["t_occ"]):
        raise ValueError("decision frames differ")
    if not all(frame in episode["valid_ends"] for episode in episodes):
        raise ValueError("decision frame does not have 25 valid actions")
    samples = [clip(episode, frame, "cuda") for episode in episodes]
    rgbd, query, truth, proprio, actions = (torch.cat(parts, dim=0)
                                               for parts in zip(*samples))
    rgb_difference = float((rgbd[0] - rgbd[1]).abs().mean())
    state_difference = float((proprio[0] - proprio[1]).abs().mean())
    config = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    seeker_args = dict(config["seeker_args"])
    seeker_args["tracker_pretrained"] = False
    tcow = expand_depth_channel(Seeker(logging.getLogger("tcow"), **seeker_args))
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    model = restore_joint_model(tcow, saved).cuda().eval()
    fixed_noise = model.flow.sample_noise(1, "cpu", torch.Generator().manual_seed(0))
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        model._latent = None
        mask_logits, _ = model.tcow(rgbd, query)
        latent = model._latent
        estimate = model.flow.sample(latent, proprio, fixed_noise.cuda().repeat(2, 1, 1))
    prediction = estimate.float().cpu().numpy()
    prediction[:, :, :5] = np.clip(prediction[:, :, :5], -1, 1)
    prediction[:, :, 5] = (prediction[:, :, 5] >= .5).astype(np.float32)
    expert = actions.cpu().numpy()
    masks = (mask_logits[:, 2, -1].float() > 0).cpu().numpy()
    output = {"seed": metadata[0]["seed"], "frame": frame,
              "targets": [meta["target_object_id"] for meta in metadata],
              "covers": [meta["correct_cover_body"] for meta in metadata],
              "rgbd_mean_abs_difference": rgb_difference,
              "proprio_mean_abs_difference": state_difference,
              "query_mask_pixel_difference": int((query[0] != query[1]).sum()),
              "predicted_container_iou_between_queries": float(
                  np.logical_and(*masks).sum() / max(np.logical_or(*masks).sum(), 1)),
              "predicted_action_difference": float(np.abs(prediction[0] - prediction[1]).mean()),
              "expert_action_difference": float(np.abs(expert[0] - expert[1]).mean()),
              "predicted_first_actions": prediction[:, 0].tolist(),
              "expert_first_actions": expert[:, 0].tolist()}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
