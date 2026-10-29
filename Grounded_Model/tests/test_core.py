"""Core regression suite: observation geometry, SE(3), gradients and retained predictions."""

import gc
import json
import unittest
import numpy as np
import torch
from unified_geometry.paths import DATA, ROOT, setup
from unified_geometry.geometry.projection import backproject
from unified_geometry.data.graph import graph


class CoreTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_projection_and_graph_correspondence(self):
        K = np.array([[2.0, 0, 1.0], [0, 2.0, 1.0], [0, 0, 1.0]])
        T = np.eye(4)
        T[:3, 3] = [0.1, 0.2, 0.3]
        points, uv = backproject(np.ones((3, 3), np.float32), K, T)
        expected = (
            np.c_[
                (uv[:, 0].astype(float) - 1) / 2,
                (uv[:, 1].astype(float) - 1) / 2,
                np.ones(9),
            ]
            + T[:3, 3]
        )
        np.testing.assert_allclose(points, expected, atol=1e-7)
        v, u = np.mgrid[:8, :8]
        uv = np.c_[u.ravel(), v.ravel()]
        order = np.random.default_rng(175).permutation(len(uv))
        inverse = np.argsort(order)
        np.testing.assert_array_equal(inverse[graph(uv)[order]], graph(uv[order]))

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is required for equivariant BF16 regression"
    )
    def test_se3_permutation_and_bf16_gradients(self):
        from unified_geometry.models.grounder import EquivariantGrounder
        from unified_geometry.losses.emd import emd
        from equivariance import equivariance

        setup()
        model = EquivariantGrounder().cuda()
        P = torch.rand(1, 4096, 3, device="cuda") * 0.4
        F = torch.randn(1, 4096, 1024, device="cuda", dtype=torch.float16)
        G = torch.rand(1, 4, 128, 3, device="cuda") * 0.4
        v, u = np.mgrid[:64, :64]
        index = torch.tensor(
            graph(np.c_[u.ravel(), v.ravel()])[None], device="cuda", dtype=torch.long
        )
        result = equivariance(model, P, F, G, index)
        self.assertLess(result["point_permutation_max_mm"], 0.05)
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.all_tasks(P, F, index)
        loss = emd(prediction.flatten(0, 1), G.flatten(0, 1)).mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        )

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for retained DINO/model integration",
    )
    def test_retained_dino_checkpoint_prediction(self):
        from unified_geometry.data.vision import Features, WEIGHTS
        from unified_geometry.models.grounder import EquivariantGrounder

        files = [
            DATA / "data/train/00000.npz",
            ROOT / "train/best.pt",
            ROOT / "evaluation/train_E178_best.npz",
            WEIGHTS / "model.safetensors",
        ]
        if not all(p.is_file() for p in files):
            self.skipTest("Retained E178 dataset/checkpoint/DINO weights are not installed")
        setup()
        cfg = json.loads((DATA / "data_config.json").read_text())
        with np.load(files[0]) as z:
            sample = {k: z[k] for k in z.files}
        extractor = Features(
            np.array(cfg["intrinsics"]), np.array(cfg["camera_to_base"]), "block24"
        )
        F = extractor([sample])
        del extractor
        gc.collect()
        torch.cuda.empty_cache()
        model = EquivariantGrounder().cuda().eval()
        saved = torch.load(files[1], map_location="cpu", weights_only=False)
        self.assertEqual(saved["step"], 12000)
        model.load_state_dict(saved["model"], strict=True)
        P = torch.tensor(sample["scene_points"][None], device="cuda")
        index = torch.tensor(graph(sample["pixel_uv"])[None], device="cuda", dtype=torch.long)
        with torch.inference_mode():
            prediction = model.all_tasks(P, F, index).cpu().numpy()[0]
        with np.load(files[2]) as z:
            expected = z["prediction"][0]
        self.assertLess(float(np.linalg.norm(prediction - expected, axis=-1).max() * 1000), 0.05)


if __name__ == "__main__":
    unittest.main()
