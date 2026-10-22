# Portions adapted from Hugging Face LeRobot SmolVLA.
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared context decoder and six-coordinate FM expert for prior/reach policies."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

SOURCE_REPO = "lerobot/smolvla_base"
SOURCE_REVISION = "d9f33c94a60fb382c90dea2164c96845bd955e28"


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
        return self.weight * value.to(x.dtype)


def apply_rotary_embedding(x, positions):
    """LeRobot apply_rope, with [B,L,H,64] input."""
    fraction = torch.arange(32, device=x.device, dtype=torch.float32) / 32
    phase = positions[..., None].float() / (10000 ** fraction)
    cos, sin = phase.cos()[..., None, :], phase.sin()[..., None, :]
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(x.dtype)


class ActionAttention(nn.Module):
    def __init__(self, cross):
        super().__init__()
        self.q_proj = nn.Linear(720, 960, bias=False)
        self.k_proj = nn.Linear(320 if cross else 720, 320, bias=False)
        self.v_proj = nn.Linear(320 if cross else 720, 320, bias=False)
        self.o_proj = nn.Linear(960, 720, bias=False)


class ActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(720, 2048, bias=False)
        self.up_proj = nn.Linear(720, 2048, bias=False)
        self.down_proj = nn.Linear(2048, 720, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ActionTransformerBlock(nn.Module):
    def __init__(self, cross):
        super().__init__()
        self.self_attn = ActionAttention(cross)
        self.mlp = ActionMLP()
        self.input_layernorm = RMSNorm(720)
        self.post_attention_layernorm = RMSNorm(720)

    def forward(self, x, positions, context_kv=None):
        residual = x
        value = self.input_layernorm(x)
        batch, length, _ = value.shape
        attn = self.self_attn
        q = apply_rotary_embedding(attn.q_proj(value).reshape(batch, length, 15, 64), positions)
        if context_kv is None:
            k = apply_rotary_embedding(attn.k_proj(value).reshape(batch, length, 5, 64), positions)
            v = attn.v_proj(value).reshape(batch, length, 5, 64)
        else:
            prefix_k, prefix_v = context_kv
            k = attn.k_proj(prefix_k.flatten(2)).reshape(batch, -1, 5, 64)
            v = attn.v_proj(prefix_v.flatten(2)).reshape(batch, -1, 5, 64)
        # Every action reads all action tokens in SA and all visual/state tokens
        # in CA, retaining the existing policy's dense interaction.
        k = k.repeat_interleave(3, dim=2)
        v = v.repeat_interleave(3, dim=2)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, length, 960)
        x = residual + attn.o_proj(attended)
        return x + self.mlp(self.post_attention_layernorm(x))


class ContextKVProjection(nn.Module):
    def __init__(self):
        super().__init__()
        # These projections are loaded from the corresponding VLM CA layer.
        self.norm = RMSNorm(960)
        self.k_proj = nn.Linear(960, 320, bias=False)
        self.v_proj = nn.Linear(960, 320, bias=False)

    def forward(self, context):
        batch, length, _ = context.shape
        x = self.norm(context)
        positions = torch.arange(length, device=x.device)[None].expand(batch, -1)
        k = apply_rotary_embedding(self.k_proj(x).reshape(batch, length, 5, 64), positions)
        v = self.v_proj(x).reshape(batch, length, 5, 64)
        return k, v


class ContextTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(960)
        self.post_attention_layernorm = RMSNorm(960)
        self.self_attn = nn.Module()
        for name, out in (("q_proj", 960), ("k_proj", 320), ("v_proj", 320), ("o_proj", 960)):
            setattr(self.self_attn, name, nn.Linear(960, out, bias=False))
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(960, 2560, bias=False)
        self.mlp.up_proj = nn.Linear(960, 2560, bias=False)
        self.mlp.down_proj = nn.Linear(2560, 960, bias=False)

    def forward(self, x, positions):
        value = self.input_layernorm(x)
        batch, length, _ = value.shape
        q = apply_rotary_embedding(self.self_attn.q_proj(value).reshape(batch, length, 15, 64), positions)
        k = apply_rotary_embedding(self.self_attn.k_proj(value).reshape(batch, length, 5, 64), positions)
        v = self.self_attn.v_proj(value).reshape(batch, length, 5, 64)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.repeat_interleave(3, dim=2).transpose(1, 2),
            v.repeat_interleave(3, dim=2).transpose(1, 2), is_causal=False)
        x = x + self.self_attn.o_proj(attended.transpose(1, 2).reshape(batch, length, 960))
        value = self.post_attention_layernorm(x)
        return x + self.mlp.down_proj(F.silu(self.mlp.gate_proj(value)) * self.mlp.up_proj(value))


class ContextDecoder(nn.Module):
    """Four transformer blocks contextualizing observation tokens."""

    def __init__(self, tokens=365, gradient_checkpointing=True):
        super().__init__()
        self.tokens = tokens
        self.gradient_checkpointing = gradient_checkpointing
        self.layers = nn.ModuleList([ContextTransformerBlock() for _ in range(4)])

    def forward(self, context, projections=None):
        if context.shape[1:] != (self.tokens, 960) or (projections is not None and len(projections) != 8):
            raise ValueError(f"context decoder requires {self.tokens} tokens and eight K/V projections")
        positions = torch.arange(self.tokens, device=context.device)[None].expand(len(context), -1)
        # All tokens describe observations available now, so they can attend
        # bidirectionally. Actions and future observations never enter here.
        for layer in self.layers:
            # ACT fits in memory; recomputation is only needed by the FM path.
            context = (checkpoint(layer, context, positions, use_reentrant=False)
                       if self.gradient_checkpointing and projections is not None and self.training and torch.is_grad_enabled()
                       else layer(context, positions))
        return context, [] if projections is None else [projection(context) for projection in projections]


class FlowMatchingHead(nn.Module):
    """Flow action expert with visual and proprioceptive conditioning."""

    flow_type = "dense_action_expert"
    horizon = 25
    action_representation = "absolute_joint"
    action_dim = 6

    def __init__(self, horizon=25):
        super().__init__()
        if horizon < 1:
            raise ValueError("positive action horizon required")
        self.horizon = horizon
        self.visual = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256),
                                    nn.SiLU(), nn.Linear(256, 960))
        self.context_tokens = 301
        self.proprio = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, 256),
                                     nn.SiLU(), nn.Linear(256, 32))
        # Start the new proprio MLP as a residual on padded, normalized state.
        nn.init.zeros_(self.proprio[-1].weight)
        nn.init.zeros_(self.proprio[-1].bias)
        self.state_proj = nn.Linear(32, 960)
        self.action_in_proj = nn.Linear(self.action_dim, 720)
        self.action_out_proj = nn.Linear(720, self.action_dim)
        self.action_time_mlp_in = nn.Linear(1440, 720)
        self.action_time_mlp_out = nn.Linear(720, 720)
        self.layers = nn.ModuleList([ActionTransformerBlock(cross=i % 2 == 1) for i in range(16)])
        self.norm = RMSNorm(720)
        self.context_projections = nn.ModuleList([ContextKVProjection() for _ in range(8)])
        for name in ("state_mean", "action_mean"):
            self.register_buffer(name, torch.zeros(self.action_dim if name.startswith("action") else 6))
        for name in ("state_std", "action_std"):
            self.register_buffer(name, torch.ones(self.action_dim if name.startswith("action") else 6))

    @property
    def output(self):
        return self.action_out_proj

    def load_pretrained(self, path: Path):
        from safetensors.torch import load_file
        state = load_file(str(path))
        expected = {key for key in self.state_dict()
                    if not key.startswith(("visual.", "proprio.", "state_mean", "state_std",
                                           "action_mean", "action_std"))}
        if state.keys() != expected:
            raise ValueError(f"action expert export keys differ: missing={sorted(expected - state.keys())}, "
                             f"extra={sorted(state.keys() - expected)}")
        if (state["action_in_proj.weight"].shape != (720, 32)
                or state["action_out_proj.weight"].shape != (32, 720)
                or state["action_out_proj.bias"].shape != (32,)):
            raise ValueError("expected SmolVLA export with 32 action coordinates")
        # Preserve pretrained weights for the six coordinates used by NexArm.
        state["action_in_proj.weight"] = state["action_in_proj.weight"][:, :self.action_dim].contiguous()
        state["action_out_proj.weight"] = state["action_out_proj.weight"][:self.action_dim].contiguous()
        state["action_out_proj.bias"] = state["action_out_proj.bias"][:self.action_dim].contiguous()
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or set(result.missing_keys) != self.state_dict().keys() - expected:
            raise ValueError("unexpected action expert load result")
        return {"source": SOURCE_REPO, "revision": SOURCE_REVISION,
                "loaded_tensors": len(state), "loaded_parameters": sum(x.numel() for x in state.values()),
                "new_parameters": sum(v.numel() for k, v in self.named_parameters() if k not in expected),
                "blocks": 8, "expert_layers": 16, "width": 720,
                "action_dim": self.action_dim, "pretrained_action_dim": 32,
                "action_projection_initialization": "first six input columns and output rows; no action padding",
                "context_shape": [self.context_tokens, 960],
                "interaction": "bidirectional SA + dense CA"}

    def conditioning(self, latent, proprio):
        if latent.shape[1:] != (300, 768) or proprio.shape[1:] != (6,):
            raise ValueError("expected visual tokens [B,300,768] and proprio [B,6]")
        state = (proprio - self.state_mean) / self.state_std
        padded = F.pad(state, (0, 26)) + self.proprio(state)
        context = torch.cat((self.visual(latent), self.state_proj(padded)[:, None]), dim=1)
        return context, [projection(context) for projection in self.context_projections]

    def set_statistics(self, statistics):
        if statistics.get("action_representation") != "absolute_joint":
            raise ValueError("absolute joint normalization required")
        for name in ("state_mean", "state_std", "action_mean", "action_std"):
            value = torch.as_tensor(statistics[name], dtype=torch.float32)
            expected = self.action_dim if name.startswith("action") else 6
            if value.shape != (expected,) or not torch.isfinite(value).all():
                raise ValueError(f"invalid normalization statistic: {name}")
            if name.endswith("std") and (value <= 0).any():
                raise ValueError(f"nonpositive normalization statistic: {name}")
            setattr(self, name, value.to(getattr(self, name).device).clone())

    def velocity(self, noisy_action, time, context_kv):
        if noisy_action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"action expert requires [B,{self.horizon},6] noisy actions")
        embedded = self.action_in_proj(noisy_action)
        fraction = torch.linspace(0, 1, 360, device=time.device,
                                  dtype=torch.float32 if time.device.type == "mps" else torch.float64)
        periods = .004 * (4 / .004) ** fraction
        # Public API keeps noise at time=0; pretrained SmolVLA uses noise at 1.
        phase = (1 - time.to(periods.dtype))[:, None] * (2 * math.pi / periods)[None]
        time_emb = torch.cat((phase.sin(), phase.cos()), dim=-1).to(embedded.dtype)
        x = torch.cat((embedded, time_emb[:, None].expand_as(embedded)), dim=-1)
        x = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x)))
        positions = torch.arange(self.horizon, device=x.device)[None].expand(len(x), -1)
        for index, kv in enumerate(context_kv):
            x = self.layers[2 * index](x, positions)
            x = self.layers[2 * index + 1](x, positions, kv)
        return -self.action_out_proj(self.norm(x))

    def forward(self, latent, proprio, noisy_action, time):
        context, kv = self.conditioning(latent, proprio)
        return self.velocity(noisy_action, time, kv), context

    def sample_noise(self, batch, device, generator=None):
        return torch.randn((batch, self.horizon, self.action_dim), generator=generator, device=device)

    def training_path(self, action):
        if action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"expected H{self.horizon} actions with {self.action_dim} coordinates")
        normalized = (action - self.action_mean) / self.action_std
        noise = torch.randn_like(normalized)
        smol_time = torch.distributions.Beta(1.5, 1.0).sample((len(action),)).to(action.device) * .999 + .001
        time = 1 - smol_time
        noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * normalized
        return noisy, time, normalized - noise

    def sample(self, latent, proprio, noise, steps=10):
        _, kv = self.conditioning(latent, proprio)
        return self.sample_actions(noise, kv, steps)

    @torch.no_grad()
    def sample_actions(self, noise, context_kv, steps=10):
        if steps < 1:
            raise ValueError("positive flow steps required")
        if noise.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"action noise must have shape [B,{self.horizon},6]")
        estimate = noise.float().clone()
        for index in range(steps):
            fraction = index / steps
            time = torch.full((len(noise),), fraction, device=noise.device)
            velocity = self.velocity(estimate, time, context_kv).float()
            estimate += velocity / steps
        return estimate * self.action_std + self.action_mean
