"""Regression checks for validation coverage, padding and phase-specific errors."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from memory_occlusion.experiments.tcow_joint_flow.validation import closed_loop_rows, evaluate


def fixture():
    n = 43
    valid = np.zeros(n, bool)
    valid[10:37] = True
    action = np.zeros((n, 6), np.float32)
    action[:, 5] = 1
    action[12:17, 5] = action[31:35, 5] = 0
    episode = dict(rgb=np.zeros((n, 240, 320, 3), np.uint8),
                   depth=np.ones((n, 240, 320), np.float32),
                   mask=np.zeros((n, 3, 240, 320), bool),
                   query=np.ones((240, 320), bool),
                   proprio=np.tile(np.arange(n)[:, None], (1, 6)).astype(np.float32),
                   action=action, action_valid=valid, events=(10, 20, 30), name="fixture")
    meta = {"phase_intervals": [
        {"phase": "cover_close", "start_frame": 12, "end_frame_exclusive": 17},
        {"phase": "object_close", "start_frame": 31, "end_frame_exclusive": 35}]}
    calls = []
    model = SimpleNamespace(eval=lambda: None)

    def tracker(rgbd, query):
        model._latent = torch.zeros((1, 300, 768))
        return torch.zeros((1, 3, 30, 1, 1)), None

    def sample(latent, proprio, noise):
        end = int(proprio[0, 0])
        calls.append(end)
        prediction = torch.arange(1, 26).float()[None, :, None].expand(1, 25, 6).clone() / 100
        prediction[:, :, 5] = .75
        prediction[:, min(25, 37 - end):] = 1e6  # Must never contribute to any metric.
        return prediction

    model.tcow = tracker
    model.flow = SimpleNamespace(sample=sample,
                                 sample_noise=lambda batch, device, generator: torch.zeros(batch, 25, 32))
    return episode, meta, model, calls


def distributed_worker(rank, root):
    torch.set_num_threads(1)
    root = Path(root)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2)
    episode, _meta, model, _calls = fixture()
    rows = [{"path": ".", "seed": seed, "split": "val"} for seed in range(3)]
    with patch("memory_occlusion.experiments.tcow_joint_flow.validation.load_joint_episode", return_value=episode):
        result = evaluate(model, root, rows, "cpu", distributed=True)
        if rank == 0:
            reference = evaluate(model, root, rows, "cpu")
            for key in ("action_mae", "action_mae_first", "action_mae_first10",
                        "action_mae_per_offset", "action_mae_per_joint", "action_chunks", "action_valid_steps"):
                np.testing.assert_allclose(result[key], reference[key], rtol=1e-7)
            assert result["cover_close"] == reference["cover_close"]
            assert result["object_close"] == reference["object_close"]
    dist.destroy_process_group()


class ValidationTest(unittest.TestCase):
    def test_distributed_uneven_episode_counts(self):
        _episode, meta, _model, _calls = fixture()
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / "episode.json").write_text(json.dumps(meta))
            mp.spawn(distributed_worker, args=(temp,), nprocs=2, join=True)

    def test_every_chunk_padding_and_close_phases(self):
        episode, meta, model, calls = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "episode.json").write_text(json.dumps(meta))
            rows = [{"path": ".", "seed": 0, "split": "val"}]
            with patch("memory_occlusion.experiments.tcow_joint_flow.validation.load_joint_episode",
                       return_value=episode):
                result = evaluate(model, root, rows, "cpu")
        self.assertEqual(calls, [10, 20, 30])
        self.assertEqual(result["action_chunks"], 3)
        self.assertEqual(result["action_valid_steps"], 49)
        expected = []
        for end in (10, 20, 30):
            for offset in range(min(25, 37 - end)):
                grip = .75 if 12 <= end + offset < 17 or 31 <= end + offset < 35 else .25
                expected.append((offset, ((offset + 1) / 100 * 5 + grip) / 6))
        self.assertAlmostEqual(result["action_mae"], np.mean([value for _, value in expected]), places=6)
        self.assertAlmostEqual(result["action_mae_first"], .05, places=6)
        self.assertAlmostEqual(result["action_mae_first10"],
                               np.mean([v for offset, v in expected if offset < 10]), places=6)
        for offset in range(25):
            self.assertAlmostEqual(result["action_mae_per_offset"][offset],
                                   np.mean([v for i, v in expected if i == offset]), places=6)
        self.assertEqual(result["cover_close"], {"gripper_mae": .75, "gripper_error_rate": 1., "valid_steps": 5})
        self.assertEqual(result["object_close"], {"gripper_mae": .75, "gripper_error_rate": 1., "valid_steps": 12})

    def test_test_split_rejected_and_scene_selection_fixed(self):
        rows = [{"target": target, "seed": seed, "path": f"{seed}_{target}", "split": "val"}
                for seed in range(4) for target in ("Milk", "Butter", "Yogurt", "Popcorn")]
        selected = closed_loop_rows(rows)
        self.assertEqual(selected, closed_loop_rows(list(reversed(rows))))
        self.assertEqual(len({r["seed"] for r in selected}), 4)
        with self.assertRaises(ValueError):
            closed_loop_rows([{**rows[0], "split": "test"}])
        with self.assertRaises(ValueError):
            evaluate(None, Path("."), [{**rows[0], "split": "test"}], "cpu")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
