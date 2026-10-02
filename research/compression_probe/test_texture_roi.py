import unittest
import numpy as np
from prepare_texture_roi import prepare_roi


class TextureROITests(unittest.TestCase):
    def test_physical_tumor_margin(self):
        labels = np.ones((25, 25, 25), dtype=np.uint8)
        labels[12, 12, 12] = 2
        roi, report = prepare_roi(labels, (1, 1, 4))
        self.assertFalse(roi[17, 12, 12])  # Exactly5mm excluded.
        self.assertTrue(roi[18, 12, 12])
        self.assertFalse(roi[12, 12, 13])  #4mm excluded.
        self.assertTrue(roi[12, 12, 14])   #8mm retained.
        self.assertFalse(roi[12, 12, 12])
        self.assertEqual(report['tumor_voxels'], 1)

    def test_negative_case_and_image_edge(self):
        labels = np.ones((9, 9, 9), dtype=np.uint8)
        roi, report = prepare_roi(labels, (1, 1, 1))
        self.assertEqual(int(roi.sum()), 125)
        self.assertEqual(report['tumor_voxels'], 0)
        self.assertFalse(report['duct_mask_available'])

    def test_empty_and_invalid(self):
        labels = np.zeros((5, 5, 5), dtype=np.uint8)
        roi, report = prepare_roi(labels, (1, 1, 1))
        self.assertFalse(roi.any())
        self.assertEqual(report['status'], 'empty-do-not-extract')
        for spacing in ((0, 1, 1), (1, 1), (float('nan'), 1, 1)):
            with self.assertRaises(ValueError):
                prepare_roi(labels, spacing)
        for bad in (np.full((5, 5, 5), .5), np.full((5, 5, 5), 28)):
            with self.assertRaises(ValueError):
                prepare_roi(bad, (1, 1, 1))
        with self.assertRaises(ValueError):
            prepare_roi(labels, (1, 1, 1), tumor_margin_mm=-1)


if __name__ == '__main__':
    unittest.main()
