import unittest

import torch

from conditional_nested_denoiser import ConditionalNestedDenoiser
from diffusion_contracts import cumulative_alphas, noisy_sample, epsilon_ddim_step, contract_trajectory


class DiffusionContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.alphas = cumulative_alphas(torch.linspace(0.0001, 0.02, 1000, dtype=torch.float64))

    def test_known_noise_reconstructs_clean(self):
        clean, noise = torch.randn(3, 4, 7, 9, dtype=torch.float64), torch.randn(3, 4, 7, 9, dtype=torch.float64)
        for t in (0, 300, 999):
            sample = noisy_sample(clean, noise, self.alphas, torch.full((3,), t, dtype=torch.long))
            reconstructed = epsilon_ddim_step(sample, noise, float(self.alphas[t]), 1.)
            torch.testing.assert_close(clean, reconstructed, rtol=1e-10, atol=1e-10)

    def test_all_depth_pruning_contracts(self):
        for dense in (True, False):
            model = ConditionalNestedDenoiser(channels=(8, 8, 16, 16), time_dim=16, context_dim=16, dense=dense)
            x, context = torch.randn(2, 8, 17, 19), torch.randn(2, 5, 16)
            outputs = model(x, 7, context, all_heads=True)
            for head, expected in enumerate(outputs, 1):
                torch.testing.assert_close(expected, model(x, 7, context, head=head).sample, rtol=0, atol=0)

    def test_deterministic_trajectory_does_not_mutate_inputs(self):
        model = ConditionalNestedDenoiser(channels=(8, 16), context_dim=16, time_dim=16).eval()
        noise, source, context = torch.randn(2, 4, 9, 11), torch.randn(2, 4, 9, 11), torch.randn(2, 5, 16)
        originals = [x.clone() for x in (noise, source, context)]
        rng = torch.get_rng_state().clone()
        output = contract_trajectory(model, noise, source, context, self.alphas, (999, 499, 0), (1, 1, 1))
        repeated = contract_trajectory(model, noise, source, context, self.alphas, (999, 499, 0), (1, 1, 1))
        torch.testing.assert_close(output, repeated, rtol=0, atol=0)
        self.assertEqual(output.shape, noise.shape)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for x, original in zip((noise, source, context), originals):
            torch.testing.assert_close(x, original, rtol=0, atol=0)
        for times, heads in (((0, 499), (1, 1)), ((999, 999), (1, 1)), ((999,), (2,)), ((999,), ())):
            with self.assertRaises(ValueError):
                contract_trajectory(model, noise, source, context, self.alphas, times, heads)

    def test_invalid_schedule(self):
        for betas in (torch.tensor([0.]), torch.tensor([1.]), torch.tensor([float('nan')])):
            with self.assertRaises(ValueError):
                cumulative_alphas(betas)


if __name__ == '__main__':
    unittest.main()
