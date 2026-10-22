"""Frozen Track-On-R streaming query features, using the official pretrained model."""
from argparse import Namespace
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
from torch.nn import functional as F
import yaml

REPOSITORY = Path('/workspace/track_on')
CHECKPOINT = Path('/workspace/robotics_assets/pretrained/trackon/track_on_r.pt')
COMMIT = '04efe8a137bd76e45e07382f02748794c9b61a96'


class FrozenTrackOn(nn.Module):
    def __init__(self):
        super().__init__()
        commit = subprocess.check_output(['git', '-C', str(REPOSITORY), 'rev-parse', 'HEAD'], text=True).strip()
        assert commit == COMMIT, (commit, COMMIT)
        sys.path.insert(0, str(REPOSITORY))
        from model.trackon import Track_On2
        from utils.coord_utils import get_points_on_a_grid
        self.grid = get_points_on_a_grid
        args = Namespace(**yaml.safe_load((REPOSITORY / 'config/test.yaml').read_text()))
        args.grad_checkpoint = False
        self.model = Track_On2(args)
        state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True, mmap=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        expected = {k for k in self.model.state_dict() if k.startswith('backbone.vit_encoder.dinov3.')}
        assert set(missing) == expected and not unexpected, (missing, unexpected)
        self.model.memory_extension(args.M_i)
        self.requires_grad_(False); self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def features(self, frame):
        assert frame.shape == (1, 3, 240, 320)
        return self.model.extract_frame_features(frame)

    @torch.no_grad()
    def initialize(self, features, click):
        # Match the official Predictor: one target plus its fixed 20x20 support grid.
        support = self.grid(20, (240, 320), features[-1].device).squeeze(0)
        queries = torch.cat((click.reshape(1, 2), support))
        coordinates = queries.clone()
        coordinates[:, 0] = coordinates[:, 0] / 320
        coordinates[:, 1] = coordinates[:, 1] / 240
        coordinates = coordinates * 2 - 1
        fused = features[-1].reshape(1, self.model.Hf, self.model.Wf, 256).permute(0, 3, 1, 2)
        initial = F.grid_sample(fused, coordinates[None, :, None], align_corners=False,
                                padding_mode='border').squeeze(0).squeeze(-1).T.contiguous()
        return dict(initial=initial, memory=initial.new_zeros(len(queries), self.model.M, 256),
                    mask=torch.ones(len(queries), self.model.M, device=initial.device, dtype=torch.bool))

    @torch.no_grad()
    def step(self, features, state):
        # Same update as official Predictor.forward_frame; additionally expose q_new.
        position, logit, hidden = self.model.track_frame(
            state['initial'], state['mask'], state['memory'], features, 240, 320)
        state['memory'] = torch.roll(state['memory'], -1, 1)
        state['memory'][:, -1] = hidden
        state['mask'] = torch.roll(state['mask'], -1, 1)
        state['mask'][:, -1] = False
        return hidden[0], position[0], logit[0].sigmoid()

    @staticmethod
    def clone_state(state):
        return {key: value.clone() for key, value in state.items()}
