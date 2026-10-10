"""Regression checks for six-coordinate FM and post-shuffle task identities."""
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
import torch

from controllers.nexarm_mujoco_backend import JOINT_NAMES, MUJOCO_JOINTS
from OATFlow.dataset.episode import context, setup_context_scene
from OATFlow.environment.env import SCENE_XML, MemoryOcclusionEnv, configure_mvp_model
from OATFlow.policy.model import FlowMatchingHead
from OATFlow.policy.loader import split_training_batches
from OATFlow.policy.data import tracking_starts
from OATFlow.policy.evaluate_tcow import foreground_iou, high_iou
from OATFlow.policy.train_tcow import balanced_training_subset
from OATFlow.policy.features import FeatureDataset, feature_shard
from OATFlow.task import TARGETS, TASKS, cover_assignment_from_state, task_id_from_state


class ActionFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.flow = FlowMatchingHead()

    def test_training_path_has_only_supervised_coordinates(self):
        action = torch.randn(2, 25, 6)
        noisy, time, truth = self.flow.training_path(action)
        self.assertEqual(noisy.shape, action.shape)
        self.assertEqual(self.flow.sample_noise(2, "cpu").shape, action.shape)
        torch.testing.assert_close(noisy + (1 - time[:, None, None]) * truth, action)

    def test_euler_updates_all_six_coordinates_without_mutating_noise(self):
        noise = self.flow.sample_noise(1, "cpu")
        original = noise.clone()
        with patch.object(self.flow, "velocity", side_effect=lambda x, t, kv: torch.ones_like(x)):
            first = self.flow.sample_actions(noise, [], steps=4)
            second = self.flow.sample_actions(noise, [], steps=4)
        torch.testing.assert_close(first, noise + 1)
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(noise, original, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "25,6"):
            self.flow.sample_actions(torch.randn(1, 25, 32), [])

    def test_real_velocity_supervises_every_output_row(self):
        noisy = self.flow.sample_noise(1, "cpu").requires_grad_()
        context = torch.randn(1, 7, 960)
        context_kv = [projection(context) for projection in self.flow.context_projections]
        velocity = self.flow.velocity(noisy, torch.tensor([.4]), context_kv)
        self.assertEqual(velocity.shape, (1, 25, 6))
        self.assertEqual(self.flow.action_in_proj.weight.shape, (720, 6))
        self.assertEqual(self.flow.action_out_proj.weight.shape, (6, 720))
        velocity.square().mean().backward()
        self.assertTrue(torch.all(self.flow.action_out_proj.weight.grad.abs().sum(1) > 0))
        self.assertTrue(torch.isfinite(noisy.grad).all())
        self.assertTrue(torch.all(noisy.grad.abs().sum((0, 1)) > 0))
        self.flow.zero_grad(set_to_none=True)


class TrackingSubsetTests(unittest.TestCase):
    def test_subset_preserves_tasks_targets_and_seed_separation(self):
        rows = [dict(task_id=task, target_id=target, seed=task * 100 + target * 10 + demo,
                     split="train", group="standard")
                for task in range(12) for target in (0, 1) for demo in range(8)]
        selected = balanced_training_subset(rows, 10, 0)
        self.assertEqual(len(selected), 120)
        self.assertEqual(selected, balanced_training_subset(rows, 10, 0))
        self.assertEqual(len({r['seed'] for r in selected}), 120)
        for task in range(12):
            for target in (0, 1):
                self.assertEqual(sum(r['task_id'] == task and r['target_id'] == target for r in selected), 5)
        with self.assertRaisesRegex(ValueError, "even"):
            balanced_training_subset(rows, 5, 0)


class TrackingIoUTests(unittest.TestCase):
    def test_empty_masks_do_not_raise_foreground_iou(self):
        truth = torch.zeros((1, 3, 2, 2), dtype=torch.bool)
        truth[0, 0, 0, 0] = True
        score, positive = foreground_iou(truth.clone(), truth)
        self.assertEqual(float(score[0, 0]), 1.)
        self.assertTrue(torch.isnan(score[0, 1:]).all())
        self.assertEqual(positive.tolist(), [[True, False, False]])
        missed, _ = foreground_iou(torch.zeros_like(truth), truth)
        self.assertEqual(float(missed[0, 0]), 0.)

    def test_stop_requires_hidden_target_and_no_absent_mask_hallucination(self):
        result = dict(foreground_iou=dict(target=.9, occluder=.9, container=.9),
                      occluded_target_iou=.4, post_shuffle_cover_iou=dict(occluder=.9, container=.9),
                      absent_false_positive_pixel_rate=dict(occluder=0., container=0.))
        self.assertFalse(high_iou(result, .85))
        result['occluded_target_iou'] = .9
        result['absent_false_positive_pixel_rate']['occluder'] = .006
        self.assertFalse(high_iou(result, .85))
        result['absent_false_positive_pixel_rate']['occluder'] = 0.
        self.assertTrue(high_iou(result, .85))


class AccumulationTests(unittest.TestCase):
    def test_tracking_sampling_stays_uniform_in_contact_phases(self):
        self.assertEqual(tracking_starts(32), list(range(32)))
        self.assertEqual(tracking_starts(1), [0])

    def test_short_action_chunks_keep_global_valid_step_weighting(self):
        rows = [{"frames": 25}, {"frames": 50}]
        samples = [(0, 5, True), (0, 23, True), (1, 1, False), (1, 5, True)]
        physical, weights = split_training_batches([[0, 1, 2, 3]], samples, rows, 2)
        self.assertEqual(physical, [[0, 1], [2, 3]])
        self.assertEqual(weights, [(4, 45, True, False, 2), (4, 45, False, True, 4)])
        x, y = torch.randn(4, 25, 6), torch.randn(4, 25, 6)
        valid = torch.arange(25)[None] < torch.tensor([19, 1, 0, 25])[:, None]
        parameter = torch.tensor(.4, requires_grad=True)
        full_loss = (((parameter * x - y).square()) * valid[:, :, None]).sum() / (45 * 6)
        full_loss.backward()
        expected = parameter.grad.clone()
        parameter.grad = None
        for micro in physical:
            partial = (((parameter * x[micro] - y[micro]).square()) * valid[micro, :, None]).sum() / (45 * 6)
            partial.backward()
        torch.testing.assert_close(parameter.grad, expected)


class FeatureCacheTests(unittest.TestCase):
    def test_shuffled_samples_keep_frame_actions_and_terminal_validity(self):
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / "features.pt")
            shard = dict(frame=torch.tensor([10, 23]),
                         latent=torch.randn(2, 300, 768, dtype=torch.bfloat16),
                         wrist=torch.randn(2, 64, 768, dtype=torch.bfloat16),
                         proprio=torch.randn(2, 6), action=torch.randn(2, 25, 6),
                         valid=torch.arange(25)[None] < torch.tensor([25, 1])[:, None])
            torch.save(shard, path)
            dataset = FeatureDataset(dict(episodes=[dict(path="scene", file=path)]),
                                     [dict(path="scene")], [(0, 23, True), (0, 10, True), (0, 11, True)])
            for item, frame_index in ((0, 1), (1, 0)):
                for actual, name in zip(dataset[item], ("latent", "proprio", "action", "valid", "wrist")):
                    torch.testing.assert_close(actual, shard[name][frame_index], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "missing cached frame"):
                dataset[2]
            feature_shard.cache_clear()


class RLResumeTests(unittest.TestCase):
    def test_lr_override_preserves_noise_lr_and_optimizer_moments(self):
        from OATFlow.rl.train import restore_actor_optimizer

        actor, noise = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
        source = torch.optim.AdamW([dict(params=[actor], lr=3e-8),
                                   dict(params=[noise], lr=1e-6)])
        (actor.square().sum() + noise.square().sum()).backward()
        source.step()
        saved = source.state_dict()
        for override in (None, 3e-7):
            with self.subTest(lr=override):
                optimizer = torch.optim.AdamW([dict(params=[actor], lr=.1),
                                               dict(params=[noise], lr=.2)])
                restore_actor_optimizer(optimizer, saved, override)
                self.assertEqual(optimizer.param_groups[0]["lr"], 3e-8 if override is None else override)
                self.assertEqual(optimizer.param_groups[1]["lr"], 1e-6)
                for parameter in (actor, noise):
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(optimizer.state[parameter][key], source.state[parameter][key])


class TaskMappingTests(unittest.TestCase):
    def make_env(self):
        env = MemoryOcclusionEnv.__new__(MemoryOcclusionEnv)
        env.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        configure_mvp_model(env.model)
        env.data = mujoco.MjData(env.model)
        env.cover_qadr = {name: env.model.joint(f"{name}_joint").qposadr[0]
                          for name in ("cover_a", "cover_b")}
        env.target_qadr = {name: env.model.joint(f"{name}_joint").qposadr[0] for name in TARGETS}
        env.robot_qpos_addresses = np.array([env.model.joint(MUJOCO_JOINTS[name]).qposadr[0]
                                            for name in JOINT_NAMES])
        return env

    def test_all_tasks_after_one_two_and_three_real_swaps(self):
        env = self.make_env()
        for task in TASKS:
            for swaps in (1, 2, 3):
                with self.subTest(task=task["task_id"], swaps=swaps):
                    _, plan = setup_context_scene(env, task["task_id"], 57000 + task["task_id"], swaps)
                    ownership = env.assignment.copy()
                    context(env, plan, lambda phase, valid: None)
                    self.assertEqual(cover_assignment_from_state(env), ownership)
                    self.assertEqual(task_id_from_state(env, task["objects"]), task["task_id"])

    def test_stale_assignment_is_rejected(self):
        env = self.make_env()
        _, plan = setup_context_scene(env, 3, 57003, 1)
        context(env, plan, lambda phase, valid: None)
        env.assignment[TARGETS[0]] = "cover_b" if env.assignment[TARGETS[0]] == "cover_a" else "cover_a"
        with self.assertRaisesRegex(ValueError, "physical containment"):
            task_id_from_state(env, TASKS[3]["objects"])


if __name__ == "__main__":
    unittest.main()
