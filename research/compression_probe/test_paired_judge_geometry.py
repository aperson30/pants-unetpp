import unittest
import nibabel as nib
import numpy as np
from paired_judge_geometry import restore_native, score_tumor_maps


class PairGeometryTests(unittest.TestCase):
    def test_axis_flip_permutation_round_trip_without_interpolation(self):
        array = np.arange(120, dtype=np.float32).reshape(4, 5, 6)
        affine = np.array([[0, -2, 0, 17], [1, 0, 0, -12], [0, 0, -3, 42], [0, 0, 0, 1]], dtype=float)
        native = nib.Nifti1Image(array, affine)
        canonical = nib.as_closest_canonical(native)
        restored = restore_native(np.asarray(canonical.dataobj), canonical, native)
        np.testing.assert_array_equal(restored, array)

    def test_shape_or_grid_change_refused(self):
        native = nib.Nifti1Image(np.zeros((4, 5, 6), dtype=np.float32), np.eye(4))
        with self.assertRaises(ValueError):
            restore_native(np.zeros((5, 4, 6)), native, native)
        moved = nib.Nifti1Image(np.zeros(native.shape), np.diag([2., 1., 1., 1.]))
        with self.assertRaises(ValueError):
            restore_native(np.zeros(native.shape), moved, native)

    def test_mask_suppression_and_remote_false_positive_separate(self):
        tumor = np.zeros((5, 5, 5), dtype=bool)
        tumor[1, 1, 1] = True
        raw = np.zeros(tumor.shape, dtype=np.float32)
        raw[tumor] = .5
        masked = np.zeros_like(raw)
        candidates = np.zeros_like(raw)
        candidates[4, 4, 4] = .9
        result = score_tumor_maps(tumor, raw, masked, candidates, 1.)
        self.assertEqual(result['tumor_mean_raw_probability'], .5)
        self.assertEqual(result['tumor_mean_masked_probability'], 0.)
        self.assertEqual(result['candidate_confidence_on_tumor'], 0.)
        self.assertGreater(result['patient_max_candidate_score'], .8)

    def test_invalid_scores_refused(self):
        tumor = np.ones((2, 3, 4), dtype=bool)
        raw = np.zeros(tumor.shape, dtype=np.float32)
        for invalid in [np.full(tumor.shape, np.nan), np.zeros((4, 3, 2))]:
            with self.assertRaises(ValueError):
                score_tumor_maps(tumor, invalid, raw, raw, 1.)


if __name__ == '__main__':
    unittest.main()
