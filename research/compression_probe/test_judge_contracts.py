import tempfile
import unittest
from pathlib import Path

import numpy as np
from judge_contracts import (crop_slices, expand_map, postprocess_probability,
                             require_same_grid, reserve_arm_directory,
                             tumor_crop_fraction)


class JudgeContracts(unittest.TestCase):
    def setUp(self):
        self.bounds = dict(x_start=2, x_finish=6, y_start=1, y_finish=4,
                           z_start=3, z_finish=5)

    def test_xyz_bounds_zyx_array_and_exclusive_finish(self):
        expected = np.arange(24).reshape(2, 3, 4) / 24
        full = expand_map(expected, (7, 8, 9), self.bounds)
        np.testing.assert_array_equal(full[3:5, 1:4, 2:6], expected.astype(np.float32))
        self.assertEqual(np.count_nonzero(full), 23)
        self.assertEqual(full[5, 1, 2], 0)

    def test_invalid_bounds_and_shapes_rejected(self):
        for key, value in [('x_start', -1), ('x_finish', 10),
                           ('z_finish', 3), ('y_start', 1.0)]:
            bounds = dict(self.bounds, **{key: value})
            with self.assertRaises(ValueError):
                crop_slices((7, 8, 9), bounds)
        with self.assertRaises(ValueError):
            expand_map(np.zeros((4, 3, 2)), (7, 8, 9), self.bounds)

    def test_physical_origin_spacing_direction_checked(self):
        affine = np.diag([.7, .7, 5., 1.])
        affine[:3, 3] = [42, -73, 20]
        require_same_grid((7, 8, 9), affine, (7, 8, 9), affine.copy())
        for index in [(0, 3), (2, 2), (0, 0)]:
            changed = affine.copy()
            changed[index] += 1
            with self.assertRaises(ValueError):
                require_same_grid((7, 8, 9), affine, (7, 8, 9), changed)
        with self.assertRaises(ValueError):
            require_same_grid((7, 8, 9), affine, (8, 7, 9), affine)

    def test_invalid_geometry_rejected(self):
        for bad in [np.zeros((4, 4)), np.full((4, 4), np.nan)]:
            with self.assertRaises(ValueError):
                require_same_grid((2, 3, 4), bad, (2, 3, 4), bad)

    def test_labels_and_voxel_dilation(self):
        for label in [1, 4, 5, 2, 3, 6, 0]:
            segmentation = np.zeros((9, 9, 9), dtype=np.uint8)
            segmentation[4, 4, 4] = label
            probability = np.full(segmentation.shape, .4, dtype=np.float32)
            before = probability.copy()
            filtered, mask = postprocess_probability(probability, segmentation)
            self.assertEqual(mask.sum(), 125 if label in [1, 4, 5] else 0)
            np.testing.assert_array_equal(probability, before)
            np.testing.assert_array_equal(filtered[mask], before[mask])

    def test_crop_exclusion_is_not_detector_error(self):
        tumor = np.zeros((7, 8, 9), dtype=bool)
        tumor[3, 1, 2] = True
        tumor[6, 1, 2] = True
        self.assertEqual(tumor_crop_fraction(tumor, self.bounds), .5)
        with self.assertRaises(ValueError):
            tumor_crop_fraction(np.zeros_like(tumor), self.bounds)

    def test_existing_arm_refused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            path = reserve_arm_directory(Path(root) / 'case' / 'original')
            marker = path / 'sentinel.txt'
            marker.write_text('preserve', encoding='utf8')
            with self.assertRaises(FileExistsError):
                reserve_arm_directory(path)
            self.assertEqual(marker.read_text(encoding='utf8'), 'preserve')

    def test_nonfinite_or_invalid_probability_rejected(self):
        for value in [np.nan, np.inf, -1, 1.1]:
            with self.assertRaises(ValueError):
                postprocess_probability(np.full((3, 3, 3), value), np.zeros((3, 3, 3)))


if __name__ == '__main__':
    unittest.main()
