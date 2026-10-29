"""Core regression suite: geometry, gradients and optional retained-model integration."""

import gc
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from sam3_pick.paths import HF_CACHE, PICK_DATA
from sam3_pick.data.target_cloud import extract_cloud
from sam3_pick.inference.fusion import filtered_points, fuse


class CoreTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_depth_cloud_and_fusion(self):
        codes = np.zeros((240, 320), np.uint16)
        codes[90:140, 120:170] = 500
        mask = codes > 0
        K = np.array([[300, 0, 159.5], [0, 300, 119.5], [0, 0, 1.0]])
        T = np.eye(4)
        cloud, good, count = extract_cloud(codes, mask, K, T, np.random.default_rng(1))
        self.assertTrue(good)
        self.assertGreaterEqual(count, 8)
        np.testing.assert_allclose(cloud[:, 2], 1.0)
        visible = filtered_points(codes, mask, K, T)
        fused, good, stats = fuse([visible, visible], np.random.default_rng(1))
        self.assertTrue(good)
        self.assertEqual(fused.shape, (128, 3))
        self.assertGreater(stats["both_view_voxels"], 0)
        _, good, _ = fuse([np.empty((0, 3), np.float32)] * 2, np.random.default_rng(1))
        self.assertFalse(good)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for BF16 model regression")
    def test_point_order_and_bf16_gradients(self):
        from sam3_pick.models.pointcloud import PointCloudReach

        torch.manual_seed(102)
        torch.set_num_threads(4)
        model = PointCloudReach().cuda().eval()
        cloud = torch.rand(1, 128, 3, device="cuda")
        state = torch.zeros(1, 6, device="cuda")
        with torch.no_grad():
            first = model.conditioning(cloud, state)
            second = model.conditioning(cloud.flip(1), state)
            for one, two in zip(first, second):
                for a, b in zip(one, two):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        model.train()
        actions = torch.zeros(1, 25, 10, device="cuda")
        mask = torch.ones(1, 25, device="cuda", dtype=torch.bool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.loss(cloud, state, actions, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        )

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is required for retained model integration"
    )
    def test_retained_reach_and_sam3(self):
        from sam3_pick.data.target_cloud import CACHE, SOURCE
        from sam3_pick.inference.prompts import REVISION

        weights = HF_CACHE / "models--facebook--sam3/snapshots" / REVISION / "model.safetensors"
        manifest = CACHE / "train_manifest.jsonl"
        if not all(p.is_file() for p in [weights, SOURCE, manifest]):
            self.skipTest("Retained dataset/E118/SAM3 weights are not installed")
        from sam3_pick.models.pointcloud import PointCloudReach
        from sam3_pick.inference.fusion import Rig, write_scene, Observation
        from sam3_pick.inference.segmentation import Sam3Masks

        row = json.loads(manifest.read_text().splitlines()[0])
        with np.load(PICK_DATA / row["path"] / "sim_state.npz") as z:
            qpos = z["initial_qpos"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.xml"
            write_scene(path)
            rig = Rig(path)
            try:
                segment = Sam3Masks()
                observation = Observation(rig, segment)
                cloud, good, _, _ = observation(qpos, row["target_id"], np.random.default_rng(133))
            finally:
                rig.close()
        self.assertTrue(good)
        self.assertEqual(segment.verification["status"], "passed")
        del segment, observation
        gc.collect()
        torch.cuda.empty_cache()
        saved = torch.load(SOURCE, map_location="cpu", weights_only=False, mmap=True)
        self.assertEqual(saved["step"], 120000)
        model = PointCloudReach().cuda().eval()
        model.load_state_dict(saved["model"], strict=True)
        with np.load(PICK_DATA / row["path"] / "observation.npz") as z:
            state = z["joint_position"][0]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            noise = model.flow.sample_noise(
                1, "cuda", torch.Generator(device="cuda").manual_seed(0)
            )
            plan = model.sample(
                torch.tensor(cloud[None], device="cuda"),
                torch.tensor(state[None], device="cuda", dtype=torch.float32),
                noise,
                steps=10,
            )
        self.assertEqual(tuple(plan.shape), (1, 25, 10))
        self.assertTrue(torch.isfinite(plan).all())


if __name__ == "__main__":
    unittest.main()
