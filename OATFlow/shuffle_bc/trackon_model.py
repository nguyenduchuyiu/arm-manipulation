"""Trainable visual context over a native Track-On query, side ViT tokens, and state."""
import torch
from torch import nn
from torch.nn import functional as F

from OATFlow.policy.model import ContextDecoder
from OATFlow.reach.model import ReachPolicy
from OATFlow.top_pick.data import SOURCE


class TrackOnContext(ReachPolicy):
    def __init__(self):
        super().__init__()
        source = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
        self.load_state_dict(source['model'], strict=True)
        del self.xyz_encoder
        self.flow.requires_grad_(False)
        self.top_projection = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 960))
        self.wrist_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 960))
        self.view_embedding = nn.Parameter(torch.randn(2, 1, 960) * .02)
        self.context_decoder = ContextDecoder(tokens=66, gradient_checkpointing=False)
        self.xyz_head = nn.Sequential(nn.LayerNorm(960), nn.Flatten(1), nn.Linear(66 * 960, 3))
        nn.init.normal_(self.xyz_head[-1].weight, std=1e-4)
        nn.init.zeros_(self.xyz_head[-1].bias)
        self.flow.context_tokens = 66
        self.flow.eval()

    def train(self, mode=True):
        super().train(mode)
        self.flow.eval()
        return self

    def context(self, top, wrist, joints, with_kv=False):
        assert top.shape[1:] == (256,) and wrist.shape[1:] == (64, 768) and joints.shape[1:] == (6,)
        state = torch.cat(((joints[:, :5] - self.flow.state_mean[:5]) / self.flow.state_std[:5],
                           -1 - 2 * joints[:, 5:] / .0255), -1)
        state = F.pad(state, (0, 26)) + self.flow.proprio(state)
        tokens = torch.cat((self.top_projection(top)[:, None] + self.view_embedding[0],
                            self.wrist_projection(wrist) + self.view_embedding[1],
                            self.flow.state_proj(state)[:, None]), 1)
        return self.context_decoder(tokens, self.flow.context_projections if with_kv else None)

    def xyz_from_features(self, top, wrist, joints):
        context, _ = self.context(top, wrist, joints)
        return self.xyz_head(context).float()

    def normalize_xyz(self, xyz):
        return (xyz - self.xyz_mean) / self.xyz_std

    @torch.no_grad()
    def sample_features(self, top, wrist, joints, noise, steps=10):
        _, kv = self.context(top, wrist, joints, with_kv=True)
        output = self.flow.sample_actions(noise, kv, steps)
        return torch.cat((output[..., :9], -.0255 * (output[..., 9:] + 1) / 2), -1)

    def context_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if not k.startswith('flow.')}

    def load_context_state(self, state):
        assert state.keys() == self.context_state().keys()
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith('flow.') for k in missing)
