import unittest
import tempfile
from pathlib import Path
import numpy as np
from contrast_probe import alter, response_metrics, shift_mask, check_output_destination


class ContrastTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((21, 21, 21), dtype=bool)
        self.mask[9:12, 9:12, 9:12] = True
        self.roi = np.ones_like(self.mask)
        self.control = np.full(self.mask.shape, 100., dtype=np.float32)

    def test_linear_operator_amplitude_and_sign(self):
        for amplitude in (-10., -20., -40., -80., 20.):
            signal = alter(self.control, self.mask, self.roi, amplitude)-self.control
            result = response_metrics(signal*0.4, self.mask, (1., 1., 1.), amplitude)
            self.assertAlmostEqual(result['ring_corrected_retention'], 0.4, places=6)
        np.testing.assert_array_equal(self.control, np.full(self.mask.shape, 100.))

    def test_shift_without_wraparound(self):
        shifted = shift_mask(self.mask)
        self.assertEqual(shifted.sum(), self.mask.sum())
        np.testing.assert_array_equal(np.argwhere(shifted), np.argwhere(self.mask)+[2, 0, 0])
        with self.assertRaises(ValueError):
            shift_mask(self.mask, (20, 0, 0))

    def test_clipping_roi_and_zero_rejected(self):
        with self.assertRaises(ValueError):
            alter(self.control, self.mask, self.roi, 1000.)
        with self.assertRaises(ValueError):
            alter(self.control, self.mask, self.roi, 0.)
        self.roi[9, 9, 9] = False
        with self.assertRaises(ValueError):
            alter(self.control, self.mask, self.roi, -20.)

    def test_output_probe_cleans_only_own_file(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            keep = parent/'keep.txt'
            keep.write_text('preserve')
            check_output_destination(parent)
            self.assertEqual(list(parent.iterdir()), [keep])
            with self.assertRaises(RuntimeError):
                check_output_destination(parent/'missing')


if __name__ == '__main__':
    unittest.main()
