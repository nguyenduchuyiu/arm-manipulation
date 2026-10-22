"""Frozen top TCOW / second-camera ViT, shared context XYZ, and reach-trained FM."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vit_b_16

from OATFlow.policy.model import ContextDecoder
from OATFlow.reach.model import ReachPolicy
from OATFlow.top_pick.data import SOURCE, VISION
from OATFlow.tracking.point import PointXYZ, load_tcow
from OATFlow.tracking.pretrain_xyz import WEIGHTS


class FrozenWristViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = vit_b_16(image_size=320); self.vision.heads = nn.Identity()
        saved = torch.load(VISION, map_location='cpu', weights_only=True, mmap=True)
        saved = {k: v for k, v in saved.items() if not k.startswith('heads.')}
        pos = saved['encoder.pos_embedding']
        grid = F.interpolate(pos[:, 1:].transpose(1, 2).reshape(1, 768, 14, 14),
                             (20, 20), mode='bicubic', align_corners=True)
        saved['encoder.pos_embedding'] = torch.cat((pos[:, :1], grid.flatten(2).transpose(1, 2)), 1)
        self.vision.load_state_dict(saved, strict=True)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])
        self.requires_grad_(False); self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, rgb):
        if rgb.shape[1:] != (3, 320, 320):
            raise ValueError('second-camera RGB must be [B,3,320,320]')
        x = self.vision._process_input((rgb.float() / 255 - self.mean) / self.std)
        cls = self.vision.class_token.expand(len(x), -1, -1)
        x = self.vision.encoder(torch.cat((cls, x), 1))[:, 1:]
        return F.adaptive_avg_pool2d(x.transpose(1, 2).reshape(len(x), 768, 20, 20), (8, 8)).flatten(2).transpose(1, 2)


class ContextPolicy(ReachPolicy):
    def __init__(self):
        super().__init__()
        saved = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
        self.load_state_dict(saved['model'], strict=True)
        del self.xyz_encoder
        self.tracker = PointXYZ()
        load_tcow(self.tracker, WEIGHTS / 'tracking_backbone.pt', WEIGHTS / 'tracking_query.pt')
        self.tracker.requires_grad_(False); self.tracker.eval()
        self.wrist = FrozenWristViT()
        self.top_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 960))
        self.wrist_projection = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 960))
        self.view_embedding = nn.Parameter(torch.randn(2, 1, 960) * .02)
        self.context_decoder = ContextDecoder(tokens=365, gradient_checkpointing=False)
        self.xyz_head = nn.Sequential(nn.LayerNorm(960), nn.Flatten(1), nn.Linear(365 * 960, 3))
        nn.init.normal_(self.xyz_head[-1].weight, std=1e-4); nn.init.zeros_(self.xyz_head[-1].bias)
        self.flow.context_tokens = 365

    def train(self, mode=True):
        super().train(mode); self.tracker.eval(); self.wrist.eval()
        return self

    @torch.no_grad()
    def encode(self, top_rgb, query, wrist_rgb):
        return self.tracker.features(top_rgb, query)[:, -1], self.wrist(wrist_rgb)

    def context(self, top, wrist, joints, with_kv=False):
        if top.shape[1:] != (300, 768) or wrist.shape[1:] != (64, 768) or joints.shape[1:] != (6,):
            raise ValueError('expected top[B,300,768], wrist[B,64,768], joints[B,6]')
        state = torch.cat(((joints[:, :5] - self.flow.state_mean[:5]) / self.flow.state_std[:5],
                           -1 - 2 * joints[:, 5:] / .0255), -1)
        state = F.pad(state, (0, 26)) + self.flow.proprio(state)
        tokens = torch.cat((self.top_projection(top) + self.view_embedding[0],
                            self.wrist_projection(wrist) + self.view_embedding[1],
                            self.flow.state_proj(state)[:, None]), 1)
        return self.context_decoder(tokens, self.flow.context_projections if with_kv else None)

    def xyz_from_features(self, top, wrist, joints):
        context, _ = self.context(top, wrist, joints)
        return self.xyz_head(context).float()

    def normalize_xyz(self, xyz):
        return (xyz - self.xyz_mean) / self.xyz_std

    def loss_parts_features(self, top, wrist, joints, actions, mask, xyz):
        context, kv = self.context(top, wrist, joints, with_kv=bool(mask.any()))
        prediction = self.xyz_head(context).float()
        auxiliary = F.mse_loss(prediction, self.normalize_xyz(xyz))
        if mask.any():
            values = torch.cat((actions[..., :9], -1 - 2 * actions[..., 9:] / .0255), -1)
            noisy, time, target = self.flow.training_path(values)
            velocity = self.flow.velocity(noisy, time, kv)
            flow = ((velocity.float() - target.float()).square() * mask[..., None]).sum() / (mask.sum() * 10)
        else:
            flow = auxiliary.new_zeros(())
        return flow, auxiliary

    @torch.no_grad()
    def sample_features(self, top, wrist, joints, noise, steps=10):
        _, kv = self.context(top, wrist, joints, with_kv=True)
        output = self.flow.sample_actions(noise, kv, steps)
        return torch.cat((output[..., :9], -.0255 * (output[..., 9:] + 1) / 2), -1)

    @torch.no_grad()
    def sample(self, top_rgb, query, wrist_rgb, joints, noise, steps=10):
        return self.sample_features(*self.encode(top_rgb, query, wrist_rgb), joints, noise, steps)

    def context_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items()
                if not k.startswith(('flow.', 'tracker.', 'wrist.'))}

    def load_context_state(self, state):
        assert state.keys() == self.context_state().keys()
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith(('flow.', 'tracker.', 'wrist.')) for k in missing)
