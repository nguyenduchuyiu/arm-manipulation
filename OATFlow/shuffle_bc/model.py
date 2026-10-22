"""BC from tracker hidden features; XYZ is supervision only."""
import torch
from torch import nn
from torch.nn import functional as F

from OATFlow.reach.model import ReachPolicy
from OATFlow.top_pick.data import SOURCE
from OATFlow.tracking.point import PointXYZ, load_tcow

WEIGHTS = SOURCE.parents[2] / 'pretrained/vision'


class ShufflePolicy(ReachPolicy):
    def __init__(self, pretrained=True):
        super().__init__()
        if pretrained:
            saved = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
            self.load_state_dict(saved['model'], strict=True)
        del self.xyz_encoder
        self.tracker = PointXYZ()
        if pretrained:
            load_tcow(self.tracker, WEIGHTS / 'tracking_backbone.pt', WEIGHTS / 'tracking_query.pt')
        self.goal = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 960))

    def conditioning(self, hidden, joints):
        state = torch.cat(((joints[:, :5] - self.flow.state_mean[:5]) / self.flow.state_std[:5],
                           -1 - 2 * joints[:, 5:] / .0255), dim=-1)
        state = F.pad(state, (0, 26)) + self.flow.proprio(state)
        tokens = torch.stack((self.goal(hidden[:, -1]), self.flow.state_proj(state)), dim=1)
        return [projection(tokens) for projection in self.flow.context_projections]

    def loss_parts(self, rgb, query, joints, actions, mask, xyz):
        hidden = self.tracker.hidden(rgb, query)
        predicted = self.tracker.xyz[3](hidden).float()
        auxiliary = F.mse_loss(predicted, self.tracker.normalize_xyz(xyz))
        if mask.any():
            values = torch.cat((actions[..., :9], -1 - 2 * actions[..., 9:] / .0255), dim=-1)
            noisy, time, target = self.flow.training_path(values)
            velocity = self.flow.velocity(noisy, time, self.conditioning(hidden, joints))
            flow = ((velocity.float() - target.float()).square() * mask[..., None]).sum() / (mask.sum() * 10)
        else:
            # No expert action exists during shuffle: leave every FM/goal grad None.
            flow = auxiliary.new_zeros(())
        error_mm = (predicted[:, -1] * self.tracker.xyz_std + self.tracker.xyz_mean - xyz[:, -1]).norm(dim=-1).mean() * 1000
        return flow, auxiliary, error_mm

    @torch.no_grad()
    def sample(self, rgb, query, joints, noise, steps=10):
        hidden = self.tracker.hidden(rgb, query)
        output = self.flow.sample_actions(noise, self.conditioning(hidden, joints), steps)
        return torch.cat((output[..., :9], -.0255 * (output[..., 9:] + 1) / 2), dim=-1)
