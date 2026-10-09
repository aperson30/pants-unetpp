import unittest
import torch
from controlled_pilot_core import intervene, predicted_clean


class ControlledTests(unittest.TestCase):
    def test_interventions_preserve_noisy_target_and_original(self):
        x = torch.randn(1, 8, 3, 3)
        before = x.clone()
        replacement = torch.ones_like(x[:, 4:])
        for condition in ("correct", "absent", "mismatched"):
            out = intervene(x, condition, replacement)
            torch.testing.assert_close(out[:, :4], x[:, :4], rtol=0, atol=0)
        torch.testing.assert_close(x, before, rtol=0, atol=0)
        self.assertEqual(float(intervene(x, "absent")[:, 4:].abs().sum()), 0)
        torch.testing.assert_close(intervene(x, "mismatched", replacement)[:, 4:], replacement)
        with self.assertRaises(ValueError):
            intervene(x, "mismatched")

    def test_epsilon_to_clean_identity(self):
        clean, noise = torch.randn(1, 4, 3, 3), torch.randn(1, 4, 3, 3)
        alpha = torch.tensor(.3)
        noisy = alpha.sqrt()*clean + (1-alpha).sqrt()*noise
        torch.testing.assert_close(predicted_clean(noisy, noise, alpha), clean)
