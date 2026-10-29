"""16-layer action flow transformer for the joint-conditioned10D EE reach policy."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-05)
        return self.weight * value.to(x.dtype)


def apply_rotary_embedding(x, positions):
    """LeRobot apply_rope, with [B,L,H,64] input."""
    fraction = torch.arange(32, device=x.device, dtype=torch.float32) / 32
    phase = positions[..., None].float() / 10000**fraction
    cos, sin = (phase.cos()[..., None, :], phase.sin()[..., None, :])
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
        k = k.repeat_interleave(3, dim=2)
        v = v.repeat_interleave(3, dim=2)
        attended = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(batch, length, 960)
        x = residual + attn.o_proj(attended)
        return x + self.mlp(self.post_attention_layernorm(x))


class ContextKVProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(960)
        self.k_proj = nn.Linear(960, 320, bias=False)
        self.v_proj = nn.Linear(960, 320, bias=False)

    def forward(self, context):
        batch, length, _ = context.shape
        x = self.norm(context)
        positions = torch.arange(length, device=x.device)[None].expand(batch, -1)
        k = apply_rotary_embedding(self.k_proj(x).reshape(batch, length, 5, 64), positions)
        v = self.v_proj(x).reshape(batch, length, 5, 64)
        return (k, v)


class FlowMatchingHead(nn.Module):
    action_dim = 10
    action_representation = "ee_pose_delta_anchor"

    def __init__(self, horizon=25):
        super().__init__()
        self.horizon = horizon
        self.context_tokens = 2
        self.proprio = nn.Sequential(
            nn.Linear(6, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 32),
        )
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
            self.register_buffer(
                name, torch.zeros(self.action_dim if name.startswith("action") else 6)
            )
        for name in ("state_std", "action_std"):
            self.register_buffer(
                name, torch.ones(self.action_dim if name.startswith("action") else 6)
            )

    def velocity(self, noisy_action, time, context_kv):
        if noisy_action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"action expert requires [B,{self.horizon},{self.action_dim}] noisy actions"
            )
        embedded = self.action_in_proj(noisy_action)
        fraction = torch.linspace(
            0,
            1,
            360,
            device=time.device,
            dtype=torch.float32 if time.device.type == "mps" else torch.float64,
        )
        periods = 0.004 * (4 / 0.004) ** fraction
        phase = (1 - time.to(periods.dtype))[:, None] * (2 * math.pi / periods)[None]
        time_emb = torch.cat((phase.sin(), phase.cos()), dim=-1).to(embedded.dtype)
        x = torch.cat((embedded, time_emb[:, None].expand_as(embedded)), dim=-1)
        x = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(x)))
        positions = torch.arange(self.horizon, device=x.device)[None].expand(len(x), -1)
        for index, kv in enumerate(context_kv):
            x = self.layers[2 * index](x, positions)
            x = self.layers[2 * index + 1](x, positions, kv)
        return -self.action_out_proj(self.norm(x))

    def sample_noise(self, batch, device, generator=None):
        return torch.randn(
            (batch, self.horizon, self.action_dim), generator=generator, device=device
        )

    def training_path(self, action):
        if action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"expected H{self.horizon} actions with {self.action_dim} coordinates")
        normalized = (action - self.action_mean) / self.action_std
        noise = torch.randn_like(normalized)
        smol_time = (
            torch.distributions.Beta(1.5, 1.0).sample((len(action),)).to(action.device) * 0.999
            + 0.001
        )
        time = 1 - smol_time
        noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * normalized
        return (noisy, time, normalized - noise)

    @torch.no_grad()
    def sample_actions(self, noise, context_kv, steps=10):
        if steps < 1:
            raise ValueError("positive flow steps required")
        if noise.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"action noise must have shape [B,{self.horizon},{self.action_dim}]")
        estimate = noise.float().clone()
        for index in range(steps):
            fraction = index / steps
            time = torch.full((len(noise),), fraction, device=noise.device)
            velocity = self.velocity(estimate, time, context_kv).float()
            estimate += velocity / steps
        return estimate * self.action_std + self.action_mean
