"""Reach with a position-preserving, permutation-invariant target cloud encoder."""
import torch
from torch import nn
from torch.nn import functional as F

from OATFlow.reach.model import ReachPolicy


class PointCloudReach(ReachPolicy):
    def __init__(self):
        super().__init__()
        del self.xyz_encoder
        self.point_encoder = nn.Sequential(nn.Linear(3, 64), nn.SiLU(),
            nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 256), nn.SiLU())
        self.goal_projection = nn.Linear(256, 960)

    def conditioning(self, cloud_base, joints):
        if cloud_base.ndim != 3 or cloud_base.shape[-1] != 3 or joints.shape != (len(cloud_base), 6):
            raise ValueError('Expected target cloud [B,N,3] and measured joints [B,6]')
        # Dataset-global train-only normalization preserves absolute location.
        points = (cloud_base - self.xyz_mean) / self.xyz_std
        goal = self.goal_projection(self.point_encoder(points).amax(dim=1))
        state = torch.cat(((joints[:, :5]-self.flow.state_mean[:5])/self.flow.state_std[:5],
                           -1-2*joints[:, 5:]/.0255), dim=-1)
        state = F.pad(state, (0, 26)) + self.flow.proprio(state)
        context = torch.stack((goal, self.flow.state_proj(state)), dim=1)
        return [projection(context) for projection in self.flow.context_projections]
