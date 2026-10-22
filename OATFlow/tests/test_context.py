"""Observation-context token preservation and explicit no-checkpoint backward."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

from OATFlow.policy.model import ContextDecoder


class ContextTests(unittest.TestCase):
    def test_full_context_backpropagates_without_checkpointing(self):
        torch.set_num_threads(2); torch.manual_seed(108)
        decoder = ContextDecoder(tokens=5, gradient_checkpointing=False).train()
        tokens = torch.randn(1, 5, 960, requires_grad=True)
        with patch('OATFlow.policy.model.checkpoint', side_effect=AssertionError('checkpoint called')):
            context, projected = decoder(tokens, nn.ModuleList([nn.Identity() for _ in range(8)]))
            self.assertEqual(context.shape, tokens.shape)
            self.assertEqual(len(projected), 8)
            context.square().mean().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertGreater(float(tokens.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters()))


if __name__ == '__main__': unittest.main()
