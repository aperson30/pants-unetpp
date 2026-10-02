import unittest
import numpy as np
from judge_repeatability import compare


class RepeatabilityTests(unittest.TestCase):
    def test_identical_and_tumor_vs_remote_drift_separate(self):
        a = np.zeros((2, 3, 4), dtype=np.float32)
        tumor = np.zeros(a.shape, dtype=bool)
        tumor[0, 0, 0] = True
        self.assertTrue(compare(a, a, tumor)['engineering_gate_max_drift_1e4'])
        b = a.copy()
        b[1, 2, 3] = .1
        report = compare(b, a, tumor)
        self.assertFalse(report['engineering_gate_max_drift_1e4'])
        self.assertEqual(report['tumor_max_abs_probability_drift'], 0.)
        b[0, 0, 0] = .01
        self.assertGreater(compare(b, a, tumor)['tumor_mean_signed_probability_change'], 0.)

    def test_invalid_comparisons_rejected(self):
        a = np.zeros((2, 3, 4), dtype=np.float32)
        for b, tumor in ((a, np.zeros(a.shape, dtype=bool)),
                         (np.zeros((3, 2, 4)), np.ones(a.shape, dtype=bool)),
                         (np.full(a.shape, np.nan), np.ones(a.shape, dtype=bool))):
            with self.assertRaises(ValueError):
                compare(a, b, tumor)


if __name__ == '__main__':
    unittest.main()
