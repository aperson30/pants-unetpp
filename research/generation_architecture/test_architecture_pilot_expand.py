"""CPU checks for phase coverage and separation in the tiny engineering pilot."""
import unittest

import numpy as np

from architecture_pilot_expand import combine_arrays


class ExpandTests(unittest.TestCase):
    def test_pair_order_and_phase_labels(self):
        arrays = [np.full((4, 3, 512, 512), value, dtype=np.float32)
                  for value in (0, .1, .2, .3, .4, .5)]
        result = combine_arrays(arrays[:2], arrays[2:4], arrays[4:])
        np.testing.assert_array_equal(result["train_phase_index"], [0]*4 + [1]*4)
        np.testing.assert_array_equal(result["development_phase_index"], [0]*4)
        self.assertEqual(float(result["train_source"][0, 0, 0, 0]), 0)
        self.assertAlmostEqual(float(result["train_source"][4, 0, 0, 0]), .2)
        self.assertAlmostEqual(float(result["development_target"][0, 0, 0, 0]), .5)
        self.assertFalse(np.shares_memory(result["train_source"], arrays[4]))

    def test_reject_invalid_inputs(self):
        valid = np.zeros((4, 3, 512, 512), dtype=np.float32)
        for invalid in (valid.astype(np.float64), valid[:3], valid + 2,
                        np.full_like(valid, np.nan)):
            with self.assertRaises(ValueError):
                combine_arrays((invalid, valid), (valid, valid), (valid, valid))


if __name__ == "__main__":
    unittest.main()
