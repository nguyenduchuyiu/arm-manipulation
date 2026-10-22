"""Pick-only RL reset, terminal and variable-length transition invariants."""
import unittest

import mujoco
import numpy as np
import torch

from OATFlow.pick.scene import scene_spec
from OATFlow.rl.env import GAMMA, LAMBDA, PickEnv
from OATFlow.rl.model import NoiseHead


class PickRLTests(unittest.TestCase):
    def make_env(self, frames=300):
        env = PickEnv(scene_spec(550000), execute=15, max_frames=frames)
        self.addCleanup(env.close)
        return env

    def test_queries_share_physical_reset(self):
        env = self.make_env()
        first = None
        for target in range(4):
            packet = env.reset(seed=7100000, task_id=0, target_id=target)
            state = (env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy(), packet['proprio'])
            if first is not None:
                for actual, expected in zip(state, first):
                    np.testing.assert_array_equal(actual, expected)
            first = state
            self.assertEqual(packet['target_id'], target)
            self.assertEqual(packet['task_id'], 0)

    def test_timeout_reports_actual_length_and_bootstrap_boundary(self):
        env = self.make_env(frames=16)
        packet = env.reset(seed=7100000, task_id=0, target_id=0)
        hold = np.tile(packet['proprio'], (15, 1))
        _, first = env.step_chunk(hold)
        self.assertEqual(first['frames'], 15)
        self.assertFalse(first['terminated'])
        self.assertFalse(first['truncated'])
        _, final = env.step_chunk(hold)
        self.assertEqual(final['frames'], 1)
        self.assertFalse(final['terminated'])
        self.assertTrue(final['truncated'])
        self.assertAlmostEqual(final['discount'], GAMMA)
        self.assertAlmostEqual(final['trace_discount'], GAMMA * LAMBDA)
        self.assertEqual(final['episode']['reason'], 'timeout')
        self.assertAlmostEqual(env.data.time, 16 / 25)

    def test_wrong_lift_terminates_without_target_success(self):
        env = self.make_env()
        packet = env.reset(seed=7100000, task_id=0, target_id=1)
        env.data.qpos[env.target_qadr['RedCube'] + 2] += .1
        mujoco.mj_forward(env.model, env.data)
        _, result = env.step_chunk(np.tile(packet['proprio'], (15, 1)))
        self.assertTrue(result['terminated'])
        self.assertFalse(result['truncated'])
        self.assertFalse(result['episode']['success'])
        self.assertEqual(result['episode']['wrong_object'], 'RedCube')
        self.assertEqual(result['episode']['reason'], 'wrong_object_lifted')
        self.assertEqual(result['frames'], 1)

    def test_h50_noise_initialization_and_bounds(self):
        head = NoiseHead(50)
        std = head(torch.randn(3, 960), torch.randn(3, 50, 6), torch.zeros(3))
        self.assertEqual(std.shape, (3, 50, 6))
        torch.testing.assert_close(std, torch.full_like(std, .02))
        self.assertTrue(((std > .005) & (std < .08)).all())


if __name__ == '__main__':
    unittest.main()
