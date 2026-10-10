"""ReinFlow Gaussian denoising paths; the BC observation stack stays frozen.

Method: https://reinflow.github.io/ (Zhang et al., NeurIPS 2025).
The PPO action is the whole latent Markov path, not a Gaussian fitted to the
deterministic final waypoint. Initial noise is parameter independent, so its
log density cancels in the policy ratio.
"""
import math

import numpy as np
import torch
from torch import nn

from OATFlow.prior.model import PriorPolicy


FLOW_STEPS = 10
VELOCITY_MODULES = ("action_in_proj", "action_out_proj", "action_time_mlp_in",
                    "action_time_mlp_out", "layers", "norm")


def load_prior(path, device="cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if saved["architecture"] != "oracle_task_target_fm" or saved["config"]["action_representation"] != "absolute_joint":
        raise ValueError("six-action absolute-joint prior required")
    policy = PriorPolicy().to(device).eval()
    policy.load_state_dict(saved["model"], strict=True)
    policy.requires_grad_(False)
    for name in VELOCITY_MODULES:
        getattr(policy.flow, name).requires_grad_(True)
    return policy, saved


class NoiseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(960 + 150 + 1, 128), nn.SiLU(), nn.Linear(128, 150))
        # Small initial transition noise in standardized action coordinates.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log((.02 - .005) / (.08 - .02)))

    def forward(self, context, noisy, time):
        value = torch.cat((context.float(), noisy.flatten(1), time[:, None]), 1)
        return (.005 + .075 * self.net(value).sigmoid()).reshape(-1, 25, 6)


class Critic(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return self.net(state).squeeze(-1)


@torch.no_grad()
def encode(policy, packets):
    views = [torch.as_tensor(np.stack([p[name] for p in packets]), device="cuda").permute(0, 3, 1, 2)
             for name in ("overview", "wrist")]
    proprio = torch.as_tensor(np.stack([p["proprio"] for p in packets]), device="cuda")
    task = torch.tensor([p["task_id"] for p in packets], device="cuda")
    target = torch.tensor([p["target_id"] for p in packets], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        context, kv = policy.conditioning(*views, proprio, task, target)
    # Cache the complete frozen conditioning, including proprio/task tokens.
    return context.mean(1).float(), [(k.float(), v.float()) for k, v in kv]


def distribution(policy, head, pooled, kv, noisy, time):
    # Keep the Gaussian means in float32: bf16 rounding can dominate a small
    # exploration sigma and corrupt the PPO likelihood ratio.
    velocity = policy.flow.velocity(noisy, time, kv).float()
    mean = noisy + velocity / FLOW_STEPS
    std = head(pooled, noisy, time)
    return torch.distributions.Normal(mean, std)


@torch.no_grad()
def sample(policy, head, pooled, kv, generator):
    batch = len(pooled)
    noisy = torch.randn((batch, 25, 6), device="cuda", generator=generator)
    chain, logprob = [noisy], torch.zeros(batch, device="cuda")
    for index in range(FLOW_STEPS):
        time = torch.full((batch,), index / FLOW_STEPS, device="cuda")
        dist = distribution(policy, head, pooled, kv, noisy, time)
        noisy = dist.mean + dist.stddev * torch.randn(noisy.shape, device="cuda", generator=generator)
        logprob += dist.log_prob(noisy).sum((1, 2))
        chain.append(noisy)
    action = noisy * policy.flow.action_std + policy.flow.action_mean
    if not torch.isfinite(action).all():
        raise ValueError("nonfinite FM action")
    return action, torch.stack(chain, 1), logprob


def log_probability(policy, head, pooled, kv, chain):
    batch = len(chain)
    noisy = chain[:, :-1].reshape(batch * FLOW_STEPS, 25, 6)
    next_noisy = chain[:, 1:].reshape(batch * FLOW_STEPS, 25, 6)
    time = torch.arange(FLOW_STEPS, device="cuda", dtype=torch.float32).repeat(batch) / FLOW_STEPS
    expanded_kv = [(k.repeat_interleave(FLOW_STEPS, 0), v.repeat_interleave(FLOW_STEPS, 0)) for k, v in kv]
    dist = distribution(policy, head, pooled.repeat_interleave(FLOW_STEPS, 0), expanded_kv, noisy, time)
    logprob = dist.log_prob(next_noisy).sum((1, 2)).reshape(batch, FLOW_STEPS).sum(1)
    entropy = dist.entropy().mean((1, 2)).reshape(batch, FLOW_STEPS).mean(1)
    return logprob, entropy, dist.stddev.mean()
