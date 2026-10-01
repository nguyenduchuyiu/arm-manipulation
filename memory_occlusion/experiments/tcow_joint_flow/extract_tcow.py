"""Export TCOW alone from an end-to-end checkpoint, excluding all FM weights."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from memory_occlusion.experiments.tcow_joint_flow.model import expand_depth_channel
from memory_occlusion.experiments.tcow_joint_flow.tcow import Seeker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joint-checkpoint", type=Path, required=True)
    p.add_argument("--config-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    joint = torch.load(args.joint_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    state = joint["model"]
    tcow = {key.removeprefix("tcow."): tensor.contiguous()
            for key, tensor in state.items() if key.startswith("tcow.")}
    flow = [key for key in state if key.startswith("flow.")]
    if not tcow or not flow or len(tcow) + len(flow) != len(state):
        raise ValueError("joint checkpoint does not contain only tcow.* and flow.* weights")
    seeker_args = dict(config["seeker_args"])
    seeker_args["tracker_pretrained"] = False
    model = expand_depth_channel(Seeker(logging.getLogger("tcow"), **seeker_args))
    model.load_state_dict(tcow, strict=True)
    payload = {"net_seeker": tcow, "seeker_args": seeker_args,
               "input_channels": 5, "source_joint_checkpoint": str(args.joint_checkpoint),
               "source_step": joint["step"],
               "note": "TCOW only; all flow.* weights excluded"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    restored = torch.load(args.output, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(restored["net_seeker"], strict=True)
    print(json.dumps({"output": str(args.output), "source_step": joint["step"],
                      "tcow_tensors": len(tcow), "excluded_flow_tensors": len(flow),
                      "parameters": sum(t.numel() for t in tcow.values())}), flush=True)


if __name__ == "__main__":
    main()
