import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from region_metrics import load_same_grid, measure


class RegionMetricsTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.zeros((15, 15, 15), dtype=np.uint8)
        self.labels[6:9, 6:9, 6:9] = 28
        self.image = np.full(self.labels.shape, 100, dtype=np.float32)
        self.image[self.labels == 28] = 50

    def test_identity(self):
        report = measure(self.image, self.image, self.labels, [1, 1, 2])
        self.assertEqual(report['lesions'][0]['volume_mm3'], 54)
        self.assertEqual(report['lesions'][0]['mae_hu'], 0)
        self.assertEqual(report['lesions'][0]['contrast_ratio'], 1)
        self.assertIsNone(report['lesions'][0]['reference_cnr'])

    def test_erased_lesion(self):
        reconstruction = np.full(self.labels.shape, 100, dtype=np.float32)
        row = measure(self.image, reconstruction, self.labels, [1, 1, 1])['lesions'][0]
        self.assertEqual(row['mae_hu'], 50)
        self.assertEqual(row['contrast_ratio'], 0)

    def test_msd_label_two(self):
        labels = np.where(self.labels == 28, 2, 0)
        report = measure(self.image, self.image, labels, [1]*3, tumor_label=2)
        self.assertEqual(report['tumor_label'], 2)
        self.assertEqual(report['tumor_voxels'], 27)

    def test_negative_and_six_connectivity(self):
        empty = np.zeros_like(self.labels)
        self.assertEqual(measure(self.image, self.image, empty, [1]*3)['lesions'], [])
        empty[6, 6, 6] = empty[7, 7, 7] = 28
        self.assertEqual(len(measure(self.image, self.image, empty, [1]*3)['lesions']), 2)

    def test_bad_inputs(self):
        for spacing in ([0, 1, 1], [float('nan'), 1, 1]):
            with self.assertRaises(ValueError):
                measure(self.image, self.image, self.labels, spacing)
        bad = self.image.copy(); bad[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            measure(self.image, bad, self.labels, [1]*3)
        with self.assertRaises(ValueError):
            measure(self.image, self.image, self.labels.astype(float) + 0.5, [1]*3)

    def test_mismatched_grid_and_units(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'{i}.nii.gz' for i in range(3)]
            for path in paths:
                im = nib.Nifti1Image(self.image, np.eye(4)); im.header.set_xyzt_units('mm'); nib.save(im, path)
            load_same_grid(paths)
            affine = np.eye(4); affine[0, 3] = 1
            im = nib.Nifti1Image(self.image, affine); im.header.set_xyzt_units('mm'); nib.save(im, paths[-1])
            with self.assertRaises(ValueError): load_same_grid(paths)
            nib.save(nib.Nifti1Image(self.image, np.eye(4)), paths[-1])
            with self.assertRaises(ValueError): load_same_grid(paths)


if __name__ == '__main__':
    unittest.main()
