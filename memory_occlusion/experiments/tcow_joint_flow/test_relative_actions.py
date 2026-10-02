"""Check chunk anchors, absolute gripper, padding and inverse action decoding."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from memory_occlusion.dataset.prepare_relative_actions import relative_chunks
from memory_occlusion.experiments.tcow_joint_flow.data import training_clip
from memory_occlusion.experiments.tcow_joint_flow.flow_matching import DenseFlowMatching
from memory_occlusion.experiments.tcow_joint_flow.test_validation import fixture


class RelativeActionsTest(unittest.TestCase):
    def test_roundtrip_anchor_gripper_and_padding(self):
        episode, _meta, _model, _calls = fixture()
        state = episode["proprio"] / 50
        action = episode["action"].copy()
        action[:, :5] = np.arange(len(action))[:, None] / 60
        chunks = relative_chunks(action, state, episode["action_valid"], 10, "test")
        self.assertEqual(chunks["starts"].tolist(), [10, 20, 30])
        self.assertEqual(chunks["valid"].sum(axis=1).tolist(), [25, 17, 7])
        for index, start in enumerate(chunks["starts"]):
            length = int(chunks["valid"][index].sum())
            restored = chunks["action"][index, :length].copy()
            restored[:, :5] += state[start, :5]
            np.testing.assert_allclose(restored, action[start:start + length], atol=1e-7)
            np.testing.assert_array_equal(chunks["action"][index, :length, 5], action[start:start + length, 5])
        episode.update(proprio=state, action=action, action_representation="relative_joint",
                       relative_action=chunks["action"], relative_starts=chunks["starts"],
                       relative_valid=chunks["valid"], relative_anchor=chunks["anchor"])
        _rgbd, _query, _masks, observed, target, valid = training_clip(episode, 30, True, "cpu")
        np.testing.assert_array_equal(target[0].numpy(), chunks["action"][-1])
        self.assertEqual(int(valid.sum()), 7)
        np.testing.assert_array_equal(observed[0].numpy(), state[30])

    def test_sample_restores_fixed_anchor_only_on_arm(self):
        flow = SimpleNamespace(relative_actions=True,
                               action_mean=torch.tensor([.1, .2, .3, .4, .5, .75]),
                               action_std=torch.ones(6),
                               conditioning=lambda latent, state: (None, None),
                               velocity=lambda estimate, time, kv: torch.zeros_like(estimate))
        state = torch.tensor([[.8, .7, .6, .5, .4, 0.]])
        result = DenseFlowMatching.sample(flow, None, state, torch.zeros(1, 25, 32))
        torch.testing.assert_close(result[:, :, :5], (flow.action_mean[:5] + state[0, :5])[None, None].expand(1, 25, 5))
        torch.testing.assert_close(result[:, :, 5], torch.full((1, 25), .75))
        flow.relative_actions = False
        absolute = DenseFlowMatching.sample(flow, None, state, torch.zeros(1, 25, 32))
        torch.testing.assert_close(absolute, flow.action_mean[None, None].expand(1, 25, 6))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
