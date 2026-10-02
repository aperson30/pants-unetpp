import unittest
import numpy as np
from contrast_sensitivity import contrasts


class ContrastTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.ones((21, 21, 21), dtype=np.uint8)
        self.labels[9:12, 9:12, 9:12] = 2
        self.image = np.full(self.labels.shape, 100., dtype=np.float32)
        self.image[self.labels == 2] = 50.

    def test_identity_and_global_shift_preserve_contrast(self):
        for delta in (0, 20):
            result = contrasts(self.image, self.image + delta, self.labels, (1, 1, 1))
            lesion = result['lesions'][0]
            self.assertEqual(lesion['lesion_mean_shift_hu'], delta)
            self.assertEqual(len(lesion['axial_slices']), 3)
            for row in lesion['rings']:
                self.assertEqual(row['signed_contrast_ratio'], 1)
                self.assertEqual(row['ring_mean_shift_hu'], delta)

    def test_erasure_zeroes_contrast(self):
        reconstructed = np.full_like(self.image, 100.)
        result = contrasts(self.image, reconstructed, self.labels, (1, 1, 1))
        for row in result['lesions'][0]['rings']:
            self.assertEqual(row['signed_contrast_ratio'], 0)

    def test_empty_pancreas_ring_and_invalid_inputs(self):
        labels = self.labels.copy()
        labels[labels == 1] = 0
        result = contrasts(self.image, self.image, labels, (1, 1, 1))
        for row in result['lesions'][0]['rings']:
            if row['tissue'] == 'gt_pancreas_only':
                self.assertEqual(row['voxels'], 0)
                self.assertNotIn('signed_contrast_ratio', row)
        with self.assertRaises(ValueError):
            contrasts(self.image, self.image, labels, (0, 1, 1))
        with self.assertRaises(ValueError):
            contrasts(self.image, self.image[:, :, :1], labels, (1, 1, 1))


if __name__ == '__main__':
    unittest.main()
