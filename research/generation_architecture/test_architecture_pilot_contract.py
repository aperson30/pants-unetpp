from types import SimpleNamespace
import unittest
import torch
from architecture_pilot_contract import draw_latents, validate_engineering_manifest


class PilotContracts(unittest.TestCase):
    def test_cap_and_scope(self):
        manifest = {"scope": "candidate-paired architecture-only exploratory feasibility",
                    "medical_quality_tested": False, "clinical_training_eligible": False,
                    "phase_prompts": ["An venous phase CT slice.", "An arterial phase CT slice."],
                    "training_case_ids": ["a", "b", "c", "d"], "development_case_ids": ["e", "f"],
                    "triplets_sha256": "a" * 64}
        validate_engineering_manifest(manifest, .5)
        for cap in (None, True, .1, .51, float("nan")):
            with self.assertRaises(ValueError):
                validate_engineering_manifest(manifest, cap)
        reversed_manifest = dict(manifest, phase_prompts=list(reversed(manifest["phase_prompts"])))
        with self.assertRaises(ValueError):
            validate_engineering_manifest(reversed_manifest, .5)
        manifest["development_case_ids"][0] = "a"
        with self.assertRaises(ValueError):
            validate_engineering_manifest(manifest, .5)

    def test_fresh_posterior_draws_reproducible_without_global_rng_mutation(self):
        class Scheduler:
            config = SimpleNamespace(num_train_timesteps=1000)
            def add_noise(self, target, noise, timestep):
                return target + noise
        mean, std = torch.zeros(2, 4, 8, 8), torch.ones(2, 4, 8, 8)
        stats = (mean, std, mean, std)
        context = torch.zeros(1, 2, 16)
        before = torch.get_rng_state().clone()
        first = draw_latents(stats, 0, 10, context, Scheduler())
        same = draw_latents(stats, 0, 10, context, Scheduler())
        other = draw_latents(stats, 0, 11, context, Scheduler())
        for x, y in zip(first, same):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertFalse(torch.equal(first[0], other[0]))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_fixed_timestep_preserves_shared_draws(self):
        class Scheduler:
            config = SimpleNamespace(num_train_timesteps=1000)
            def add_noise(self, target, noise, timestep):
                return target + noise*timestep[:, None, None, None]
        mean, std = torch.zeros(1, 4, 2, 2), torch.ones(1, 4, 2, 2)
        stats = (mean, std, mean, std)
        ctx = torch.zeros(1, 2, 16)
        first = draw_latents(stats, 0, 5, ctx, Scheduler(), timestep_override=100, return_target=True)
        other = draw_latents(stats, 0, 5, ctx, Scheduler(), timestep_override=900, return_target=True)
        torch.testing.assert_close(first[0][:, 4:], other[0][:, 4:], rtol=0, atol=0)
        torch.testing.assert_close(first[3], other[3], rtol=0, atol=0)
        torch.testing.assert_close(first[4], other[4], rtol=0, atol=0)
        self.assertEqual(int(first[1]), 100)
        with self.assertRaises(ValueError):
            draw_latents(stats, 0, 5, ctx, Scheduler(), timestep_override=1000)
