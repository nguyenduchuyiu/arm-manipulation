"""Check live camera routing, wrist conditioning and action/mask gradient isolation."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from memory_occlusion.experiments.tcow_joint_flow.data import training_clip
from memory_occlusion.experiments.tcow_joint_flow.flow_matching import DenseFlowMatching
from memory_occlusion.experiments.tcow_joint_flow.model import JointTCOWFlow
from memory_occlusion.experiments.tcow_joint_flow.test_validation import fixture


def tracker_fixture():
    # Small spatial layer; the action expert below is the real full architecture.
    block = SimpleNamespace(norm1=nn.LayerNorm(768), attn=nn.Linear(768, 768),
                            norm2=nn.LayerNorm(768), mlp=nn.Linear(768, 768))
    source = SimpleNamespace(patch_embed=SimpleNamespace(proj=nn.Conv2d(5, 768, 16, 16),
                                                        img_size=(240, 320)),
                             cls_token=nn.Parameter(torch.zeros(1, 1, 768)),
                             pos_embed=nn.Parameter(torch.randn(1, 301, 768) * .02),
                             blocks=[block], norm=nn.LayerNorm(768))

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesformer = SimpleNamespace(model=source)

        def forward(self, rgbd):
            return torch.ones(len(rgbd), 768, 30, 15, 20), None

    class Tracker(nn.Module):
        def __init__(self):
            super().__init__()
            self.seeker = nn.Module()
            self.seeker.tracker_backbone = Backbone()
            self.head = nn.Parameter(torch.ones(()))

        def forward(self, rgbd, query):
            self.seeker.tracker_backbone(rgbd)
            return self.head * torch.ones(len(rgbd), 3, 30, 1, 1), None

    return Tracker()


class WristTest(unittest.TestCase):
    def test_current_frame_without_crop_or_future(self):
        episode, *_ = fixture()
        episode["wrist_rgb"] = np.broadcast_to(np.arange(43, dtype=np.uint8)[:, None, None, None],
                                              (43, 320, 320, 3))
        clip = training_clip(episode, 20, True, "cpu", include_wrist=True)
        self.assertEqual(clip[6].shape, (1, 3, 320, 320))
        torch.testing.assert_close(clip[6], torch.full((1, 3, 320, 320), (20 / 255 - .45) / .225))

    def test_action_reads_wrist_and_context_skips_encoder(self):
        torch.manual_seed(0)
        tracker = tracker_fixture()
        flow = DenseFlowMatching(wrist_backbone=tracker.seeker.tracker_backbone)
        flow.relative_actions = True
        model = JointTCOWFlow(tracker, flow)
        latent = torch.randn(1, 300, 768)
        proprio = torch.zeros(1, 6)
        wrist = torch.randn(1, 3, 320, 320)
        noise = flow.sample_noise(1, "cpu", torch.Generator().manual_seed(0))
        time = torch.zeros(1)
        velocity, context = flow(latent, proprio, noise, time, wrist=wrist)
        self.assertEqual(context.shape, (1, 701, 960))
        velocity.square().mean().backward()
        gradient = flow.wrist_encoder.patch.weight.grad
        self.assertTrue(torch.isfinite(gradient).all() and gradient.abs().sum() > 0)
        flow.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            original = flow.sample(latent, proprio, noise, steps=1, wrist=wrist)
            changed = flow.sample(latent, proprio, noise, steps=1, wrist=wrist.flip(-1))
        self.assertGreater(float((original - changed).abs().max()), 1e-6)
        with self.assertRaisesRegex(ValueError, "requires current wrist"):
            flow.sample(latent, proprio, noise, steps=1)
        calls = []
        handle = flow.wrist_encoder.register_forward_hook(lambda *_: calls.append(1))
        logits, velocity, _, context = model(torch.zeros(1, 4, 30, 1, 1), None, wrist=wrist)
        logits.square().mean().backward()
        handle.remove()
        self.assertFalse(calls)
        self.assertIsNone(velocity)
        self.assertIsNone(context)
        self.assertTrue(all(p.grad is None for p in flow.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
