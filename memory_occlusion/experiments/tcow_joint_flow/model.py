"""Original TCOW with dense current-frame tokens and a conditional flow policy."""
from __future__ import annotations

import math

import torch
from torch import nn


class FlowPolicy(nn.Module):
    flow_type = "dense"
    def __init__(self, width: int = 256, horizon: int = 25):
        super().__init__()
        self.horizon = horizon
        self.visual = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.proprio = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, width))
        self.action = nn.Linear(6, width)
        self.action_position = nn.Parameter(torch.randn(1, horizon, width) * .02)
        self.action_time_mlp_in = nn.Linear(width * 2, width)
        self.action_time_mlp_out = nn.Linear(width, width)
        self.blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(width, 8, 1024, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(3)
        ])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 6)

    def forward(self, latent: torch.Tensor, proprio: torch.Tensor,
                noisy_action: torch.Tensor, time: torch.Tensor):
        if latent.shape[1:] != (300, 768) or proprio.shape[1:] != (6,):
            raise ValueError(f"expected [B,300,768] and [B,6], got {latent.shape}, {proprio.shape}")
        context = torch.cat((self.visual(latent), self.proprio(proprio)[:, None]), dim=1)
        half = self.action.out_features // 2
        time_dtype = torch.float32 if time.device.type == "mps" else torch.float64
        fraction = torch.linspace(0, 1, half, device=time.device, dtype=time_dtype)
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        # Our path runs noise -> action; SmolVLA's time runs action -> noise.
        phase = (1 - time.to(time_dtype))[:, None] * (2 * math.pi / period)[None]
        action_embedding = self.action(noisy_action)
        time_embedding = torch.cat((phase.sin(), phase.cos()), dim=-1).to(action_embedding.dtype)
        x = torch.cat((action_embedding, time_embedding[:, None].expand_as(action_embedding)), dim=-1)
        x = self.action_time_mlp_out(torch.nn.functional.silu(self.action_time_mlp_in(x)))
        x = x + self.action_position
        for block in self.blocks:
            x = block(x, context)
        return self.output(self.norm(x)), context

    def sample(self, latent: torch.Tensor, proprio: torch.Tensor,
               noise: torch.Tensor, steps: int = 8):
        estimate = noise.float()
        for step in range(steps):
            time = torch.full((len(noise),), (step + .5) / steps, device=noise.device)
            velocity, _ = self(latent, proprio, estimate, time)
            estimate = estimate + velocity.float() / steps
        return estimate

    def sample_noise(self, batch, device, generator=None):
        return torch.randn((batch, self.horizon, 6), generator=generator, device=device)

    def training_path(self, action):
        noise = torch.randn_like(action)
        smol_time = torch.distributions.Beta(1.5, 1.0).sample((len(action),)).to(action.device) * .999 + .001
        time = 1 - smol_time
        noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * action
        return noisy, time, action - noise


class JointTCOWFlow(nn.Module):
    def __init__(self, seeker: nn.Module, flow: nn.Module | None = None):
        super().__init__()
        self.tcow = seeker
        self.flow = FlowPolicy() if flow is None else flow
        self._latent = None
        self.tcow.seeker.tracker_backbone.register_forward_hook(self._capture_backbone)

    def _capture_backbone(self, _module, _inputs, output):
        dense = output[0]
        self._latent = dense[:, :, -1].flatten(2).transpose(1, 2)

    def forward(self, rgbd, query, proprio=None, noisy_action=None, time=None, wrist=None):
        self._latent = None
        mask_logits, _flags = self.tcow(rgbd, query)
        if self._latent is None:
            raise RuntimeError("TCOW backbone hook did not capture dense features")
        latent = self._latent
        if noisy_action is None:
            return mask_logits, None, latent, None
        velocity, context = (self.flow(latent, proprio, noisy_action, time) if wrist is None else
                             self.flow(latent, proprio, noisy_action, time, wrist=wrist))
        return mask_logits, velocity, latent, context


def restore_joint_model(seeker, checkpoint):
    flow_type = checkpoint.get("flow_type", "dense")
    if flow_type == "dense_action_expert":
        from memory_occlusion.experiments.tcow_joint_flow.flow_matching import DenseFlowMatching
        flow = DenseFlowMatching(
            relative_actions=checkpoint.get("action_representation") == "relative_joint",
            wrist_backbone=seeker.seeker.tracker_backbone if checkpoint.get("wrist_camera", False) else None)
    elif flow_type == "dense":
        flow = FlowPolicy()
    else:
        raise ValueError(f"unknown flow architecture: {flow_type}")
    model = JointTCOWFlow(seeker, flow)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def load_training_tracker(seeker, source, flow_only):
    if "model" in source:
        if not flow_only:
            raise ValueError("joint initialization requires frozen TCOW")
        state = {key.removeprefix("tcow."): value for key, value in source["model"].items()
                 if key.startswith("tcow.")}
        step = source["step"]
    else:
        if source.get("input_channels") != 5:
            raise ValueError("expected an RGB-D TCOW checkpoint")
        state, step = source["net_seeker"], source["source_step"]
    seeker.load_state_dict(state, strict=True)
    return step


def expand_depth_channel(seeker: nn.Module):
    """Insert depth before the original query channel without editing vendor code."""
    backbone = seeker.seeker.tracker_backbone
    patch = backbone.timesformer.model.patch_embed
    old = patch.proj
    if old.in_channels != 4:
        raise ValueError(f"expected RGB+query patch conv, got {old.in_channels} channels")
    new = nn.Conv2d(5, old.out_channels, old.kernel_size, old.stride,
                    old.padding, old.dilation, old.groups, old.bias is not None)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3].copy_(old.weight[:, :3])
        new.weight[:, 4].copy_(old.weight[:, 3])
        if old.bias is not None:
            new.bias.copy_(old.bias)
    patch.proj = new
    seeker.seeker.input_channels = 5
    backbone.Ci = 5
    return seeker
