"""E103-style visual goal token with auxiliary target-surface cloud supervision."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from OATFlow.top_pick.data import VISION
from OATFlow.top_pick.model import TopPickPolicy


ROOT = Path('/workspace/robotics_assets/experiments/e119_pick')
SOURCE = ROOT.parent/'e118_cloud/train/best.pt'
DATA = Path('/workspace/robotics_assets/datasets/pick_wide_rgbd')


class CloudPickPolicy(TopPickPolicy):
    def __init__(self, initialize=True):
        # Build the E103 visual/goal path, without loading the old XYZ reach model.
        super().__init__(pretrained=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(119)
            self.cloud_head = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128*3))
        self.register_buffer('cloud_mean', torch.zeros(3))
        self.register_buffer('cloud_std', torch.ones(3))
        if initialize:
            source = torch.load(SOURCE, map_location='cpu', weights_only=False, mmap=True)
            assert source['step'] == 120000
            self.flow.load_state_dict({k.removeprefix('flow.'): v for k, v in source['model'].items()
                                       if k.startswith('flow.')}, strict=True)
            weights = torch.load(VISION, map_location='cpu', weights_only=True, mmap=True)
            weights = {k: v for k, v in weights.items() if not k.startswith('heads.')}
            position = weights['encoder.pos_embedding']
            grid = position[:, 1:].transpose(1, 2).reshape(1, 768, 14, 14)
            grid = F.interpolate(grid, (20, 20), mode='bicubic', align_corners=True)
            weights['encoder.pos_embedding'] = torch.cat((position[:, :1], grid.flatten(2).transpose(1, 2)), dim=1)
            self.vision.load_state_dict(weights, strict=True)
        self.vision.requires_grad_(False); self.vision.eval()

    @torch.no_grad()
    def encode(self, rgb):
        if rgb.shape[1:] != (3, 240, 320):
            raise ValueError('Expected native front RGB [B,3,240,320]')
        # Letterbox preserves native camera geometry; mean-color padding normalizes to zero.
        square = (self.rgb_mean*255).expand(len(rgb), 3, 320, 320).clone()
        square[:, :, 40:280] = rgb
        return super().encode(square)

    def predicted_cloud(self, hidden):
        return self.cloud_head(hidden).float().reshape(-1, 128, 3)

    def loss_parts(self, features, joints, target_id, actions, mask, cloud, cloud_valid):
        hidden = self.visual_hidden(features, target_id)
        values = torch.cat((actions[..., :9], -1-2*actions[..., 9:]/.0255), dim=-1)
        noisy, time, velocity_target = self.flow.training_path(values)
        velocity = self.flow.velocity(noisy, time, self.context_from_hidden(hidden, joints))
        error = (velocity.float()-velocity_target.float()).square()
        flow = (error*mask[..., None]).sum()/(mask.sum()*10)
        prediction = self.predicted_cloud(hidden)
        if cloud_valid.any():
            a = prediction[cloud_valid]
            b = (cloud[cloud_valid]-self.cloud_mean)/self.cloud_std
            distance = torch.cdist(a, b).square()
            auxiliary = .5*(distance.amin(1).mean()+distance.amin(2).mean())
            with torch.no_grad():
                metric = torch.cdist(a*self.cloud_std+self.cloud_mean, cloud[cloud_valid])
                error_mm = .5*(metric.amin(1).mean()+metric.amin(2).mean())*1000
        else:
            auxiliary = prediction.sum()*0
            error_mm = prediction.new_zeros(())
        return flow, auxiliary, error_mm

    @torch.no_grad()
    def predict_cloud(self, features, target_id):
        hidden = self.visual_hidden(features, target_id)
        return self.predicted_cloud(hidden)*self.cloud_std+self.cloud_mean
