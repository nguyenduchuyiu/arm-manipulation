import unittest
import torch

from OATFlow.prior.scene_probe import SceneProbe, ProprioProbe, probe_loss


class SceneProbeTests(unittest.TestCase):
    def test_shapes_and_physical_normalization(self):
        torch.set_num_threads(1)
        head = SceneProbe()
        mean = torch.randn(5, 3)
        head.set_statistics(mean, torch.zeros(5, 3))
        logits, positions = head(torch.randn(2, 131, 960))
        self.assertEqual(logits.shape, (2, 4, 4))
        torch.testing.assert_close(positions, mean.expand(2, -1, -1))
        torch.testing.assert_close(head.position_std, torch.full((5, 3), .005))

    def test_only_readout_receives_gradient(self):
        head = SceneProbe()
        context = torch.randn(2, 131, 960)
        logits, positions = head(context)
        loss, _ = probe_loss(logits, positions, torch.tensor([[0,1,2,3],[1,2,3,0]]), torch.randn(2,5,3), head.position_std)
        loss.backward()
        self.assertIsNone(context.grad)
        self.assertGreater(float(head.color[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(head.position[-1].weight.grad.abs().sum()), 0)

    def test_proprio_control_shapes(self):
        head = ProprioProbe()
        logits, positions = head(torch.randn(3, 6), torch.tensor([0,1,2]))
        self.assertEqual(logits.shape, (3,4,4))
        self.assertEqual(positions.shape, (3,5,3))


if __name__ == '__main__':unittest.main()
