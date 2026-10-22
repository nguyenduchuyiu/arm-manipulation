"""Frozen visual tokens condition reach-initialized FM without a context decoder."""
import torch
from torch import nn
from torch.nn import functional as F

from OATFlow.reach.flow import FlowMatchingHead
from OATFlow.top_pick.data import SOURCE

FEATURE_BLOCK = 16


def pack_top(hidden):
    blocks = hidden.float().reshape(*hidden.shape[:-1], 768 // FEATURE_BLOCK, FEATURE_BLOCK)
    scale = (blocks.abs().amax(-1) / 127).half().clamp_min(1e-6)
    codes = (blocks / scale.float()[..., None]).round().clamp(-127, 127).to(torch.int8)
    return codes.reshape_as(hidden), scale


def unpack_top(codes, scale):
    blocks = codes.reshape(*codes.shape[:-1], 768 // FEATURE_BLOCK, FEATURE_BLOCK).half()
    return (blocks * scale[..., None]).reshape(*codes.shape)


def visual_tokens(hidden, view):
    # FM's existing RMSNorm already removes per-token scale. Normalize before
    # adding a fixed modality code; padding preserves every feature coordinate.
    value = hidden.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
    value = F.pad(value, (0, 192))
    value[..., 768 + view] = 1
    return value


class DirectPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow = FlowMatchingHead(horizon=25)
        source = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
        assert source['step'] == 115000
        state = {k.removeprefix('flow.'): v for k, v in source['model'].items() if k.startswith('flow.')}
        self.flow.load_state_dict(state, strict=True)
        self.flow.context_tokens = 365

    def conditioning(self, top, side, joints):
        assert top.shape[1:] == (300, 768) and side.shape[1:] == (64, 768) and joints.shape[1:] == (6,)
        state = torch.cat(((joints[:, :5] - self.flow.state_mean[:5]) / self.flow.state_std[:5],
                           -1 - 2 * joints[:, 5:] / .0255), -1)
        state = F.pad(state, (0, 26)) + self.flow.proprio(state)
        state = self.flow.state_proj(state)
        tokens = torch.cat((visual_tokens(top, 0), visual_tokens(side, 1), state[:, None]), 1)
        return [projection(tokens) for projection in self.flow.context_projections]

    def loss(self, top, side, joints, actions, mask):
        assert mask.any() and mask.dtype == torch.bool
        values = torch.cat((actions[..., :9], -1 - 2 * actions[..., 9:] / .0255), -1)
        noisy, time, target = self.flow.training_path(values)
        velocity = self.flow.velocity(noisy, time, self.conditioning(top, side, joints))
        return ((velocity.float() - target.float()).square() * mask[..., None]).sum() / (mask.sum() * 10)

    @torch.no_grad()
    def sample_features(self, top, side, joints, noise, steps=10):
        output = self.flow.sample_actions(noise, self.conditioning(top, side, joints), steps)
        return torch.cat((output[..., :9], -.0255 * (output[..., 9:] + 1) / 2), -1)
