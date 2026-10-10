"""ACT masking, inference isolation, normalization and gradient checks."""
import unittest

import torch

from OATFlow.prior.act import ACTHead
from OATFlow.prior.model import PriorPolicy


class ACTTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def test_padding_does_not_change_posterior_or_reconstruction(self):
        head = ACTHead(horizon=7).eval()
        state, action = torch.randn(2, 6), torch.randn(2, 7, 6)
        valid = torch.tensor([[True]*3+[False]*4, [True]*5+[False]*2])
        altered = action.clone().masked_fill(~valid[:, :, None], 10000.)
        mu, logvar = head.posterior(state, action, valid)
        other_mu, other_logvar = head.posterior(state, altered, valid)
        torch.testing.assert_close(mu, other_mu, atol=0, rtol=0)
        torch.testing.assert_close(logvar, other_logvar, atol=0, rtol=0)
        prediction = torch.randn_like(action)
        loss, metrics = head.loss(prediction, action, valid, mu, logvar)
        other_loss, _ = head.loss(prediction, altered, valid, mu, logvar)
        torch.testing.assert_close(loss, other_loss, atol=0, rtol=0)
        torch.testing.assert_close(metrics['l1'], (prediction-action).abs()[valid].mean())
        with self.assertRaisesRegex(ValueError, 'at least one valid'):
            head.posterior(state, action, torch.zeros_like(valid))

    def test_act_frontend_gradients_and_zero_latent_inference(self):
        model = PriorPolicy(horizon=50, action_head='act')
        self.assertFalse(hasattr(model, 'flow'))
        features = [torch.randn(2, 64, 768) for _ in range(2)]
        state = torch.randn(2, 6)
        task, target = torch.zeros(2, dtype=torch.long), torch.tensor([1, 3])
        action = torch.randn(2, 50, 6)
        valid = torch.ones(2, 50, dtype=torch.bool)
        (prediction, mu, logvar), context = model.act_forward_features(*features, state, task, target, action, valid)
        self.assertEqual(context.shape, (2, 131, 960))
        self.assertEqual(prediction.shape, (2, 50, 6))
        loss, _ = model.act.loss(prediction, action, valid, mu, logvar)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in (model.visual_projection, model.context_decoder, model.task_embedding,
                       model.target_embedding, model.act.proprio, model.act.posterior_params, model.act.output):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)
        self.assertTrue(all(p.grad is None for p in model.vision_encoder.parameters()))
        self.assertTrue((model.act.output.weight.grad.abs().sum(1)>0).all())
        model.eval()
        with torch.no_grad():
            (first, mu, logvar), _ = model.act_forward_features(*features, state, task, target)
            torch.manual_seed(999)
            (second, _, _), _ = model.act_forward_features(*features, state, task, target)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertIsNone(mu); self.assertIsNone(logvar)
        with self.assertRaisesRegex(ValueError, 'must not use future action labels'):
            model.act_forward_features(*features, state, task, target, action, valid)

    def test_normalization_and_checkpoint_restore(self):
        model = PriorPolicy(horizon=7, action_head='act').eval()
        statistics = dict(action_representation='absolute_joint', action_mean=[.1]*6,
                          action_std=[.2]*6, state_mean=[.3]*6, state_std=[.4]*6)
        model.act.set_statistics(statistics)
        restored = PriorPolicy(horizon=7, action_head='act').eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored.act.denormalize(torch.ones(1, 7, 6)), torch.full((1, 7, 6), .3))
        for name in ('state_mean', 'state_std', 'action_mean', 'action_std'):
            torch.testing.assert_close(getattr(restored.act, name), getattr(model.act, name))


if __name__ == '__main__':
    unittest.main()
