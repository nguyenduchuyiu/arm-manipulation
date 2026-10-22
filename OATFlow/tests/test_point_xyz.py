"""Point-channel semantics and causality for native one-camera TCOW XYZ."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from OATFlow.tracking.point import PointXYZ
from OATFlow.tracking.point_data import point_clip, project


class PointXYZTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.manual_seed(100)
        self.model = PointXYZ(image_size=(16, 24), patch_size=8, max_frames=5,
                              width=24, depth=4, heads=4, drop_path_rate=0., local_query=True).eval()
        self.rgb = torch.rand(2, 3, 5, 16, 24)
        self.query = torch.zeros(2, 1, 5, 16, 24)
        self.query[:, :, 0, 4, 5] = 1

    def test_prompt_selects_target_and_has_no_id_parameters(self):
        a = self.model(self.rgb, self.query)
        changed = self.query.clone(); changed[:, :, 0] = 0; changed[:, :, 0, 12, 20] = 1
        b = self.model(self.rgb, changed)
        self.assertGreater(float((a - b).abs().max()), 1e-7)
        self.assertFalse(any('identity' in name or 'target_embedding' in name for name, _ in self.model.named_parameters()))
        self.assertEqual(self.model.backbone.patch_embed.proj.in_channels, 4)

    def test_future_images_cannot_change_prefix_in_train_or_eval(self):
        for training in (False, True):
            self.model.train(training)
            a = self.model(self.rgb, self.query)
            changed = self.rgb.clone(); changed[:, :, 3:] = torch.rand_like(changed[:, :, 3:])
            b = self.model(changed, self.query)
            torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=1e-7)

    def test_future_gradients_zero_and_prefix_extension_stable(self):
        rgb = self.rgb.clone().requires_grad_(); query = self.query.clone().requires_grad_()
        full = self.model(rgb, query)
        rgb_grad, query_grad = torch.autograd.grad(full[0, 2].square().sum(), (rgb, query))
        self.assertEqual(float(rgb_grad[:, :, 3:].abs().max()), 0.)
        self.assertGreater(float(query_grad[0, :, 0].abs().sum()), 0.)
        self.assertEqual(float(rgb_grad[1].abs().max()), 0.)
        for t in (1, 3, 4):
            prefix = self.model(self.rgb[:, :, :t], self.query[:, :, :t])
            torch.testing.assert_close(prefix, full.detach()[:, :t], rtol=0, atol=2e-6)

    def test_xyz_loss_trains_query_channel_and_backbone(self):
        self.model.train(); self.model(self.rgb, self.query).square().mean().backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(float(self.model.backbone.patch_embed.proj.weight.grad[:, 3].abs().sum()), 0.)

    def test_normalization_and_invalid_query(self):
        self.model.set_xyz_statistics([.1, .2, .3], [.2, .3, .01])
        expected = self.model(self.rgb, self.query)[:, -1]
        torch.testing.assert_close(self.model.normalize_xyz(self.model.predict(self.rgb, self.query)), expected)
        for query in (self.query * 0, self.query[:, :, :1], self.query + float('nan')):
            with self.assertRaisesRegex(ValueError, 'query'):
                self.model(self.rgb, query)
        future = self.query.clone(); future[:, :, 1, 4, 5] = 1
        with self.assertRaisesRegex(ValueError, 'query'):
            self.model(self.rgb, future)

    def test_clip_keeps_frame0_dot_and_ignores_future_labels(self):
        rgb = torch.zeros(1, 2, 3, 240, 320, dtype=torch.uint8)
        episode = dict(context=dict(rgb=rgb.clone(), time=[0, 1]),
                       expert=dict(rgb=rgb.clone(), time=[2, 3]), query_uv=np.array([100., 80.]),
                       xyz=np.arange(12, dtype=np.float32).reshape(4, 3))
        first = point_clip(episode, 1)
        episode['expert']['rgb'].fill_(255); episode['xyz'][2:] += 100
        second = point_clip(episode, 1)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(first[1].shape, (1, 1, 30, 240, 320))
        self.assertEqual(int(first[1][:, :, 0].sum()), 29)
        self.assertEqual(int(first[1][:, :, 1:].count_nonzero()), 0)
        with self.assertRaisesRegex(ValueError, 'frame0'):
            point_clip(episode, 1, frames=1)

    def test_projection_accounts_for_center_crop(self):
        camera = dict(base=np.zeros(3), position=np.zeros(3), rotation=np.eye(3), focal=100.)
        np.testing.assert_allclose(project([0., 0., 1.], camera), [159.5, 119.5])
        np.testing.assert_allclose(project([.1, .2, 1.], camera), [169.5, 139.5])

    def test_reference_is_local_and_uses_only_initial_features(self):
        model = PointXYZ(image_size=(40, 40), patch_size=8, max_frames=5,
                         width=24, depth=1, heads=4, local_query=True)
        features = torch.arange(25).float()[None, None, :, None].expand(1, 5, 25, 24).clone()
        query = torch.zeros(1, 1, 5, 40, 40); query[:, :, 0, 20, 20] = 1
        reference = model.initial_reference(features, query)
        torch.testing.assert_close(reference, torch.full((1, 24), 12.))
        features[:, 1:] += 1000
        features[:, 0, :5] += 1000
        features[:, 0, 20:] += 1000
        torch.testing.assert_close(model.initial_reference(features, query), reference)
        features[:, 0, 12] += 9
        torch.testing.assert_close(model.initial_reference(features, query), reference + 1)

    def test_loader_decodes_only_top_and_crops_without_masks(self):
        from OATFlow.tracking.data import read_segment
        with TemporaryDirectory() as directory:
            root = Path(directory); segment = root / 'segment'; segment.mkdir()
            np.savez(segment / 'labels.npz', timestamp_s=np.arange(4), entity_xyz_base_m=np.zeros((4, 6, 3)))
            np.savez(segment / 'supervision.npz', phase=np.array(['reveal'] * 4))
            pixels = np.broadcast_to(np.arange(320).astype(np.uint8)[:, None, None], (320, 320, 3)).copy()
            frame = MagicMock(); frame.to_ndarray.return_value = pixels
            video = MagicMock(); video.__enter__.return_value = video; video.decode.return_value = [frame] * 4
            with patch('OATFlow.tracking.data.av.open', return_value=video) as opened:
                result = read_segment(root, 'segment', 3, views=('top',), center_crop=True)
            self.assertEqual(opened.call_count, 1)
            self.assertTrue(opened.call_args.args[0].endswith('rgb_top.mp4'))
            self.assertEqual(result['rgb'].shape, (1, 2, 3, 240, 320))
            self.assertEqual(int(result['rgb'][0, 0, 0, 0, 0]), 40)
            self.assertEqual(int(result['rgb'][0, 0, 0, -1, 0]), 279 % 256)
            self.assertIsNone(result['hidden'])


if __name__ == '__main__':
    unittest.main()
