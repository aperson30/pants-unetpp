import copy
import unittest

import torch

from conditional_plain_denoiser import ConditionalPlainDenoiser


class PlainControlContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = ConditionalPlainDenoiser(channels=(8, 16, 24), context_dim=16, time_dim=16)
        self.x = torch.randn(2, 8, 15, 17)
        self.context = torch.randn(2, 5, 16)

    def test_shape_single_head_and_conditioning(self):
        x = self.x.clone().requires_grad_()
        context = self.context.clone().requires_grad_()
        out = self.model(x, 7, context).sample
        self.assertEqual(out.shape, (2, 4, 15, 17))
        out.square().mean().backward()
        for grad in (x.grad[:, :4], x.grad[:, 4:], context.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.parameters()))
        self.assertEqual(len(self.model.decoder), 2)
        self.assertFalse(hasattr(self.model, "nodes"))
        self.assertFalse(torch.equal(out, self.model(x, 900, context).sample))

    def test_state_round_trip_and_time_batch(self):
        clone = copy.deepcopy(self.model)
        clone.load_state_dict(self.model.state_dict(), strict=True)
        out = self.model(self.x, 7, self.context).sample
        torch.testing.assert_close(out, clone(self.x, torch.tensor([7, 7]), self.context,
                                               return_dict=False)[0], rtol=0, atol=0)

    def test_invalid_contracts(self):
        for x, context in ((self.x[:, :4], self.context), (self.x, self.context[:, :, :8])):
            with self.assertRaises(ValueError):
                self.model(x, 7, context)
        with self.assertRaises(ValueError):
            self.model(self.x, -1, self.context)
        with self.assertRaises(ValueError):
            self.model(self.x, 7, self.context, head=1)


if __name__ == "__main__":
    unittest.main()
