import copy
import io
import unittest

import torch

from conditional_nested_denoiser import ConditionalNestedDenoiser, supervised_denoising_loss


class DenoiserContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = ConditionalNestedDenoiser(channels=(8, 16, 24), context_dim=16, time_dim=16)
        self.sample = torch.randn(2, 8, 15, 17)
        self.context = torch.randn(2, 5, 16)
        self.target = torch.randn(2, 4, 15, 17)

    def test_shapes_and_scalar_batch_time(self):
        heads = self.model(self.sample, torch.tensor([7, 7]), self.context, all_heads=True)
        self.assertEqual(len(heads), 2)
        for x in heads:
            self.assertEqual(x.shape, self.target.shape)
            self.assertTrue(torch.isfinite(x).all())
        torch.testing.assert_close(heads[-1], self.model(self.sample, 7, self.context).sample,
                                   rtol=0, atol=0)

    def test_pruned_head_bit_exact_output_and_gradient(self):
        reference = copy.deepcopy(self.model)
        outputs = reference(self.sample, 7, self.context, all_heads=True)
        pruned = self.model(self.sample, 7, self.context, head=1).sample
        torch.testing.assert_close(outputs[0], pruned, rtol=0, atol=0)
        outputs[0].square().mean().backward()
        pruned.square().mean().backward()
        for (name, p), (other_name, q) in zip(self.model.named_parameters(), reference.named_parameters()):
            self.assertEqual(name, other_name)
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        self.assertIsNone(self.model.nodes['2_0'].conv1.weight.grad)
        self.assertIsNone(self.model.heads['2'].weight.grad)

    def test_conditioning_reaches_every_exit(self):
        for head in (1, 2):
            x = self.sample.clone().requires_grad_()
            context = self.context.clone().requires_grad_()
            self.model.zero_grad(set_to_none=True)
            out = self.model(x, 7, context, head=head).sample
            out.square().mean().backward()
            self.assertGreater(x.grad[:, :4].abs().sum().item(), 0)
            self.assertGreater(x.grad[:, 4:].abs().sum().item(), 0)
            self.assertGreater(context.grad.abs().sum().item(), 0)
            self.assertGreater(self.model.time_mlp[0].weight.grad.abs().sum().item(), 0)
            self.assertFalse(torch.equal(out, self.model(x, 900, context, head=head).sample))

    def test_no_unrequested_heads_computed(self):
        calls = []
        hooks = [m.register_forward_hook(lambda m, a, o, key=key: calls.append(key))
                 for key, m in self.model.heads.items()]
        try:
            self.model(self.sample, 7, self.context, head=1)
            self.assertEqual(calls, ['1'])
        finally:
            for hook in hooks:
                hook.remove()

    def test_finite_update_and_exact_serialization_resume(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        def step(model, opt):
            opt.zero_grad(set_to_none=True)
            outputs = model(self.sample, 7, self.context, all_heads=True)
            loss = supervised_denoising_loss(outputs, self.target, (0.5, 0.5))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            opt.step()
            return loss.detach()
        before = self.model.heads['2'].weight.detach().clone()
        step(self.model, optimizer)
        self.assertFalse(torch.equal(before, self.model.heads['2'].weight))
        stream = io.BytesIO()
        torch.save({'model': self.model.state_dict(), 'optimizer': optimizer.state_dict()}, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        resumed = copy.deepcopy(self.model)
        resumed.load_state_dict(saved['model'], strict=True)
        resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        resumed_opt.load_state_dict(saved['optimizer'])
        torch.testing.assert_close(step(self.model, optimizer), step(resumed, resumed_opt), rtol=0, atol=0)
        for p, q in zip(self.model.parameters(), resumed.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_explicit_loss_scale(self):
        outputs = self.model(self.sample, 7, self.context, all_heads=True)
        normalized = supervised_denoising_loss(outputs, self.target, (0.5, 0.5))
        summed = supervised_denoising_loss(outputs, self.target, (1, 1))
        torch.testing.assert_close(summed, 2 * normalized)
        for weights in ((0, 0), (1,), (float('nan'), 1), (-1, 2)):
            with self.assertRaises(ValueError):
                supervised_denoising_loss(outputs, self.target, weights)

    def test_rejects_unsupported_and_malformed_inputs(self):
        for kwargs in ({'head': 0}, {'head': 3}, {'head': True},
                       {'head': 1, 'all_heads': True}, {'added_cond_kwargs': {'x': 1}}):
            with self.assertRaises(ValueError):
                self.model(self.sample, 7, self.context, **kwargs)
        with self.assertRaises(ValueError):
            self.model(self.sample[:, :4], 7, self.context)
        with self.assertRaises(ValueError):
            self.model(self.sample, torch.tensor([1, 2, 3]), self.context)
        with self.assertRaises(ValueError):
            self.model(self.sample, -1, self.context)

    def test_nondense_variant(self):
        model = ConditionalNestedDenoiser(channels=(8, 16, 24), context_dim=16, time_dim=16, dense=False)
        full = model(self.sample, 7, self.context, all_heads=True)
        torch.testing.assert_close(full[0], model(self.sample, 7, self.context, head=1).sample,
                                   rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
