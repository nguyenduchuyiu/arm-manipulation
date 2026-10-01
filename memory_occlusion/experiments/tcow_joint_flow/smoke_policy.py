"""Check pretrained import, dense conditioning and both TCOW loss phases."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from memory_occlusion.experiments.tcow_joint_flow.model import JointTCOWFlow, expand_depth_channel
from memory_occlusion.experiments.tcow_joint_flow.flow_matching import DenseFlowMatching


def grad_sum(parameter):
    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
        raise AssertionError("missing or nonfinite gradient")
    result = float(parameter.grad.float().abs().sum())
    if result == 0:
        raise AssertionError("zero gradient")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flow-weights", type=Path, required=True)
    p.add_argument("--tcow-checkpoint", type=Path)
    p.add_argument("--seeker-config", type=Path,
                   help="JSON seeker_args exported from the actual TCOW config checkpoint")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.tcow_checkpoint and not args.seeker_config:
        raise ValueError("--seeker-config is required to smoke the TCOW checkpoint")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    flow = DenseFlowMatching()
    imported = flow.load_pretrained(args.flow_weights)
    flow.set_statistics({"state_mean": [0, 0, 0, 0, 0, .5], "state_std": [.5] * 6,
                         "action_mean": [0, 0, 0, 0, 0, .5], "action_std": [.5] * 6})
    device = args.device
    flow = flow.to(device)
    report = {"device": device, "import": imported, "synthetic_smoke_only": True}
    latent = torch.randn(1, 300, 768, device=device, requires_grad=True)
    state = torch.zeros(1, 6, device=device, requires_grad=True)
    action = torch.randn(1, 25, 6, device=device) * .1
    with tqdm(total=4 if args.tcow_checkpoint else 2, desc="Memory occlusion policy smoke", unit="check",
              file=sys.stdout) as progress:
        noisy, time, truth_velocity = flow.training_path(action)
        velocity, context = flow(latent, state, noisy, time)
        assert velocity.shape == (1, 25, 6) and context.shape == (1, 301, 960)
        loss = (velocity - truth_velocity).square().mean()
        loss.backward()
        report["flow_backward"] = {"loss": float(loss.detach()), "latent_grad": grad_sum(latent),
                                   "state_grad": grad_sum(state),
                                   "expert_sa_grad": grad_sum(flow.layers[0].self_attn.q_proj.weight),
                                   "expert_ca_grad": grad_sum(flow.layers[1].self_attn.q_proj.weight),
                                   "context_projection_grad": grad_sum(flow.context_projections[0].k_proj.weight),
                                   "visual_adapter_grad": grad_sum(flow.visual[-1].weight)}
        flow.zero_grad(set_to_none=True)
        progress.update()
        with torch.inference_mode():
            noise = flow.sample_noise(1, "cpu", torch.Generator().manual_seed(0)).to(device)
            sample = flow.sample(latent.detach(), state.detach(), noise)
            changed = latent.detach().clone()
            changed[:, 149, ::2] += 2
            estimate = flow.sample(changed, state.detach(), noise)
            influence = float((sample - estimate).abs().mean())
            assert sample.shape == (1, 25, 6) and torch.isfinite(sample).all() and influence > 0
            # Check the direction/sign of the reversed pretrained velocity.
            fixed_time = torch.zeros(1, device=device)
            _, kv = flow.conditioning(latent.detach(), state.detach())
            single_step = flow.sample(latent.detach(), state.detach(), noise, steps=1)
            expected = (noise + flow.velocity(noise, fixed_time, kv))[:, :, :6]
            expected = expected * flow.action_std + flow.action_mean
            torch.testing.assert_close(single_step, expected)
            report["inference"] = {"shape": list(sample.shape), "steps": 10,
                                   "single_spatial_token_effect": influence,
                                   "finite": True, "euler_sign_and_grid": "passed"}
        progress.update()
        if args.tcow_checkpoint:
            from memory_occlusion.experiments.tcow_joint_flow.tcow import (
                Seeker, checkpoint_transformer_blocks,
            )
            config = json.loads(args.seeker_config.read_text())
            config["tracker_pretrained"] = False
            tcow = expand_depth_channel(Seeker(logging.getLogger("tcow"), **config))
            saved = torch.load(args.tcow_checkpoint, map_location="cpu", weights_only=False, mmap=True)
            if "model" in saved:
                weights = {k.removeprefix("tcow."): v for k, v in saved["model"].items()
                           if k.startswith("tcow.")}
            else:
                weights = saved["net_seeker"]
            tcow.load_state_dict(weights, strict=True)
            checkpoint_transformer_blocks(tcow)
            model = JointTCOWFlow(tcow, flow).to(device).eval()
            optimizer = torch.optim.AdamW([
                {"params": model.tcow.parameters(), "lr": 2e-5},
                {"params": model.flow.parameters(), "lr": 1e-4},
            ])
            rgbd = torch.randn(1, 4, 30, 240, 320, device=device) * .1
            query = torch.zeros(1, 1, 30, 240, 320, device=device)
            query[:, :, 0, 110:130, 150:170] = 1
            noisy, time, truth_velocity = flow.training_path(action)
            logits, velocity, features, context = model(rgbd, query, state.detach(), noisy, time)
            target = torch.zeros_like(logits)
            target[:, 0] = query[:, 0]
            mask_loss = F.binary_cross_entropy_with_logits(logits.float(), target.float())
            action_loss = (velocity.float() - truth_velocity).square().mean()
            patch = tcow.seeker.tracker_backbone.timesformer.model.patch_embed.proj.weight
            action_patch_grad = torch.autograd.grad(action_loss, patch, retain_graph=True)[0]
            assert torch.isfinite(action_patch_grad).all() and action_patch_grad.abs().sum() > 0
            (action_loss + .2 * mask_loss).backward()
            head = tcow.seeker.tracker_post_linear.weight
            report["joint_backward"] = {"tcow_source_step": saved["step"] if "model" in saved else saved["source_step"],
                                         "latent_shape": list(features.shape), "context_shape": list(context.shape),
                                         "patch_grad": grad_sum(patch), "mask_head_grad": grad_sum(head),
                                         "action_only_patch_grad": float(action_patch_grad.float().abs().sum()),
                                         "flow_head_grad": grad_sum(flow.output.weight),
                                         "action_loss": float(action_loss.detach()), "mask_loss": float(mask_loss.detach())}
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model._latent = None
            del logits, velocity, features, context, action_loss, mask_loss
            progress.update()
            calls = []
            handle = flow.register_forward_hook(lambda *_: calls.append(1))
            logits, _ = model.tcow(rgbd, query)
            F.binary_cross_entropy_with_logits(logits.float(), target.float()).backward()
            handle.remove()
            assert not calls and all(p.grad is None for p in flow.parameters())
            report["mask_only"] = {"flow_forward_calls": len(calls), "all_flow_gradients_none": True,
                                   "mask_head_grad": grad_sum(head)}
            model._latent = None
            progress.update()
    if device == "mps":
        torch.mps.synchronize()
        report["mps_driver_memory_gib"] = torch.mps.driver_allocated_memory() / 2**30
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
