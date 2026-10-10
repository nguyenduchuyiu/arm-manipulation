"""Balance and counterfactual invariants for pick pretraining."""
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from OATFlow.pick.scene import PERMUTATIONS, permutation_spec, scene_spec
from OATFlow.task import TARGETS


class PickDataTests(unittest.TestCase):
    def test_complete_permutations_balance_all_target_slots(self):
        cells = Counter((target, permutation.index(target))
                        for permutation in PERMUTATIONS for target in TARGETS)
        self.assertEqual(len(set(PERMUTATIONS)), 24)
        self.assertEqual(len(cells), 16)
        self.assertEqual(set(cells.values()), {6})

    def test_robot_pose_uses_independent_stream(self):
        seed = 550000
        robot_stream = np.random.SeedSequence(seed).spawn(2)[1]
        expected = np.random.default_rng(robot_stream).uniform(
            (-.28, -.22, -.25, -.30, -.25), (.28, .22, .25, .15, .25))
        np.testing.assert_array_equal(scene_spec(seed)["initial_arm"], expected)
        self.assertEqual(scene_spec(seed), scene_spec(seed))

    def test_permutation_nuisance_does_not_change_robot_or_slot_centers(self):
        base = scene_spec(550000)
        first, second = (permutation_spec(base, i) for i in (0, 1))
        self.assertEqual(first, permutation_spec(base, 0))
        self.assertEqual(first["initial_arm"], second["initial_arm"])
        self.assertEqual(first["centers"], second["centers"])
        self.assertNotEqual(first["positions"], second["positions"])
        self.assertNotEqual(first["yaw"], second["yaw"])


class FrozenVisionCacheTests(unittest.TestCase):
    def test_retirement_requires_exact_replacement_of_all_old_frames(self):
        from OATFlow.prior.features import retire_cached_shard
        with TemporaryDirectory() as directory:
            old, new = Path(directory) / 'old.pt', Path(directory) / 'new.pt'
            original = dict(starts=torch.tensor([1, 5]), overview=torch.tensor([10., 20.]),
                            wrist=torch.tensor([30., 40.]))
            replacement = dict(starts=torch.tensor([1, 3, 5]), overview=torch.tensor([10., 99., 21.]),
                               wrist=torch.tensor([30., 99., 40.]))
            torch.save(original, old); torch.save(replacement, new)
            with self.assertRaisesRegex(ValueError, 'changed reused feature'):
                retire_cached_shard(old, new)
            self.assertTrue(old.exists())
            replacement['overview'][-1] = 20.
            replacement['starts'][-1] = 4
            torch.save(replacement, new)
            with self.assertRaisesRegex(ValueError, 'retain every old cached frame'):
                retire_cached_shard(old, new)
            self.assertTrue(old.exists())
            replacement['starts'][-1] = 5
            torch.save(replacement, new)
            retire_cached_shard(old, new)
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())

    def test_cache_reuse_matches_frame_ids_and_keeps_missing_slots(self):
        from OATFlow.prior.features import copy_cached_frames
        cached = dict(starts=torch.tensor([1, 5, 9]),
                      overview=torch.arange(3, dtype=torch.float16)[:, None, None].expand(3, 64, 768),
                      wrist=torch.arange(10, 13, dtype=torch.float16)[:, None, None].expand(3, 64, 768))
        output = {key: torch.full((5, 64, 768), float('nan'), dtype=torch.float16)
                  for key in ('overview', 'wrist')}
        reused = copy_cached_frames(output, cached, np.array([0, 1, 6, 9, 10]))
        np.testing.assert_array_equal(reused, [False, True, False, True, False])
        for key in output:
            torch.testing.assert_close(output[key][reused], cached[key][[0, 2]], atol=0, rtol=0)
            self.assertTrue(torch.isnan(output[key][~reused]).all())
        cached['starts'] = torch.tensor([1, 1, 9])
        with self.assertRaisesRegex(ValueError, 'sorted and unique'):
            copy_cached_frames(output, cached, np.array([1, 9]))

    def test_cached_and_rgb_paths_match_and_keep_adapter_gradients(self):
        from OATFlow.prior.model import PriorPolicy
        torch.set_num_threads(2)
        torch.manual_seed(0)
        model = PriorPolicy().eval()
        images = [torch.randint(256, (1, 3, 320, 320), dtype=torch.uint8) for _ in range(2)]
        state = torch.randn(1, 6)
        task, target = torch.tensor([0]), torch.tensor([2])
        noise, time = torch.randn(1, 25, 6), torch.tensor([.4])
        features = model.encode_views(*images)
        self.assertFalse(any(value.requires_grad for value in features))
        with torch.no_grad():
            expected, _ = model(*images, state, task, target, noise, time)
        actual, _ = model.forward_features(*features, state, task, target, noise, time)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.vision_encoder.parameters()))
        for module in (model.visual_projection, model.target_embedding, model.context_decoder, model.flow):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)


class CombinatorialBatchTests(unittest.TestCase):
    def test_full_batches_cover_permutations_and_every_chunk_occurs_once(self):
        from OATFlow.pick.batching import combinatorial_batches
        plans = [[[(chunk, True) for chunk in range(15 + (episode % 7))] for episode in range(96)]]
        samples, batches, ends = combinatorial_batches(plans, 2, 256, np.random.default_rng(0))
        previous = 0
        for end in ends:
            epoch_batches = batches[previous:end]
            self.assertEqual(sorted(index for batch in epoch_batches for index in batch), list(range(len(samples))))
            for batch in epoch_batches:
                if len(batch) == 256:
                    self.assertEqual({samples[index][0] // 4 for index in batch}, set(range(24)))
            previous = end


class HorizonTests(unittest.TestCase):
    def test_h50_noise_training_velocity_and_euler(self):
        from OATFlow.policy.model import FlowMatchingHead
        torch.set_num_threads(2)
        flow = FlowMatchingHead(horizon=50)
        noise = flow.sample_noise(1, 'cpu')
        self.assertEqual(noise.shape, (1, 50, 6))
        action = torch.randn_like(noise)
        noisy, time, truth = flow.training_path(action)
        torch.testing.assert_close(noisy + (1 - time[:, None, None]) * truth, action)
        context = torch.randn(1, 7, 960)
        kv = [projection(context) for projection in flow.context_projections]
        velocity = flow.velocity(noisy, time, kv)
        self.assertEqual(velocity.shape, (1, 50, 6))
        velocity.square().mean().backward()
        self.assertTrue(torch.all(flow.action_out_proj.weight.grad.abs().sum(1) > 0))
        self.assertEqual(flow.sample_actions(noise, kv, steps=2).shape, (1, 50, 6))


if __name__ == "__main__":
    unittest.main()
