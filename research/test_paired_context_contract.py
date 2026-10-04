"""Small CPU checks for the diagnostic's intervention contract, not model quality."""
import unittest

import numpy as np

from paired_context_contract import controls


class ContractTests(unittest.TestCase):
    def test_identity_and_hu_endpoints(self):
        source = np.array([-2000, -1000, -300, 400, 1000], dtype=float)
        normalized = np.array([-1, -1, 0, 1, 1], dtype=float)
        mask = np.array([0, 1, 1, 0, 0])
        original, foreground, background = controls(source, normalized, mask)
        for result in (original, foreground, background):
            np.testing.assert_array_equal(result, normalized)

    def test_each_half_changes_only_where_requested(self):
        source = np.array([-1000, -300, 400, -300], dtype=float)
        generated = np.array([0.25, 0.5, -0.25, -0.5])
        mask = np.array([0, 1, 0, 1])
        _, foreground, background = controls(source, generated, mask)
        np.testing.assert_array_equal(foreground, [-1, 0.5, 1, -0.5])
        np.testing.assert_array_equal(background, [0.25, 0, -0.25, 0])

    def test_invalid_masks_and_geometry_fail_closed(self):
        source = np.zeros(4)
        for mask in (np.zeros(4), np.ones(4), np.array([0, 2, 0, 0]),
                     np.array([0, np.nan, 1, 0]), np.zeros(3)):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                controls(source, source, mask)
        with self.assertRaises(ValueError):
            controls(source, np.zeros(3), np.array([0, 1, 0, 0]))


if __name__ == "__main__":
    unittest.main()
