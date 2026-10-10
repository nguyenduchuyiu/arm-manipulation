"""Dense expert frames, terminal masks and cached feature/action alignment."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from OATFlow.dataset.actions import chunk_starts, save_action_chunks
from OATFlow.policy.data import sample_plan
from OATFlow.prior.features import CachedPriorDataset, shard
from OATFlow.prior.train import episode_arrays


class DenseDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(episode_arrays.cache_clear)
        self.addCleanup(shard.cache_clear)
        self.root = Path(self.directory.name)
        self.path = self.root / "episode"
        self.path.mkdir()
        self.action = np.arange(65 * 6, dtype=np.float32).reshape(65, 6)
        self.state = self.action + 1000
        self.valid = np.zeros(65, bool)
        self.valid[3:64] = True
        self.meta = dict(frames=65, decision_frames=dict(t_occ=3), task_id=7, target_id=2)
        (self.path / "episode.json").write_text(json.dumps(self.meta))
        np.savez(self.path / "supervision.npz", expert_action=self.action, action_valid=self.valid)
        np.savez(self.path / "observation.npz", joint_position=self.state)

    def test_context_and_expert_use_each_frame_once(self):
        plan = sample_plan(self.valid, 3, "episode")
        self.assertEqual(plan, [(i, i >= 3) for i in range(64)])
        broken = self.valid.copy()
        broken[20] = False
        with self.assertRaisesRegex(ValueError, "noncontiguous"):
            chunk_starts(broken, 3, "episode")
        broken = self.valid.copy()
        broken[0] = True
        with self.assertRaisesRegex(ValueError, "action boundary"):
            chunk_starts(broken, 3, "episode")

    def test_raw_horizons_and_stored_chunks_preserve_labels_and_tail_mask(self):
        stored = save_action_chunks(self.path, self.action, self.state, self.valid, self.meta)
        for horizon in (25, 50):
            arrays = episode_arrays(self.path, horizon)
            np.testing.assert_array_equal(arrays["starts"], np.arange(3, 64))
            np.testing.assert_array_equal(arrays["action"][0], self.action[3:3 + horizon])
            np.testing.assert_array_equal(arrays["joint_anchor"], self.state[3:64])
            np.testing.assert_array_equal(arrays["action"][-1], np.tile(self.action[63], (horizon, 1)))
            self.assertEqual(arrays["valid"][-1].tolist(), [True] + [False] * (horizon - 1))
            self.assertFalse(np.any(arrays["action"] == self.action[64, 0]))
            if horizon == 25:
                for key in ("starts", "action", "valid"):
                    np.testing.assert_array_equal(arrays[key], stored[key])

    def test_cached_features_follow_raw_frame_ids_and_reject_missing_frames(self):
        cache = self.root / "features.pt"
        starts = torch.arange(3, 64)
        features = dict(starts=starts, overview=starts[:, None].float(), wrist=-starts[:, None].float())
        torch.save(features, cache)
        dataset = CachedPriorDataset(self.root, [dict(path="episode")],
                                     [(0, 60, True), (0, 0, True)], {"episode": cache}, horizon=50)
        tail, first = dataset.__getitems__([0, 1])
        self.assertEqual(tail[0].item(), 63)
        self.assertEqual(first[1].item(), -3)
        np.testing.assert_array_equal(tail[2], self.state[63])
        self.assertEqual((tail[3].item(), tail[4].item()), (7, 2))
        self.assertEqual(tail[6].sum().item(), 1)
        np.testing.assert_array_equal(first[5], self.action[3:53])
        shard.cache_clear()
        torch.save({key: value[::2] for key, value in features.items()}, cache)
        with self.assertRaisesRegex(ValueError, "feature/action frame mismatch"):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
