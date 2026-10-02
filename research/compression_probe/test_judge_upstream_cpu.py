"""Run pinned publisher code, not our reimplementation, on CPU toy images.

Requires JUDGE_UPSTREAM_SOURCE pointing to independently fetched data_utils.py,
SimpleITK2.5.3 and report-guided-annotation0.3.4 in an isolated dependency path.
This is not model inference or parity with the historic released container.
"""
import hashlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

if not os.environ.get('JUDGE_UPSTREAM_SOURCE'):
    raise unittest.SkipTest('Explicit pinned upstream source/runtime required')

import numpy as np
import SimpleITK as sitk
from report_guided_annotation import extract_lesion_candidates
from judge_contracts import expand_map, postprocess_probability


def load_upstream():
    source = Path(os.environ['JUDGE_UPSTREAM_SOURCE'])
    expected = '5f09cb980619e1f459a0163e6808a1bc20188a466c000218719eff36cd1ad68a'
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Pinned upstream source digest mismatch')
    if sitk.Version_VersionString() != '2.5.3':
        raise RuntimeError('Wrong SimpleITK version')
    if importlib.metadata.version('report-guided-annotation') != '0.3.4':
        raise RuntimeError('Wrong candidate-extractor version')
    candidate_file = Path(__import__('report_guided_annotation').__file__).parent / 'extract_lesion_candidates.py'
    if hashlib.sha256(candidate_file.read_bytes()).hexdigest() != '15119b3bf469d8505c949c6c901321a0941dae340f070270f533dd15cb546990':
        raise RuntimeError('Candidate-extractor digest mismatch')
    spec = importlib.util.spec_from_file_location('pinned_panorama_data_utils', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


upstream = load_upstream()


class UpstreamCPUContracts(unittest.TestCase):
    def image(self):
        image = sitk.Image([40, 36, 20], sitk.sitkFloat32) + 35
        image.SetSpacing((.7, 1.1, 3.))
        image.SetOrigin((42., -73., 20.))
        image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
        return image

    def test_nifti_round_trip_preserves_physical_grid(self):
        image = self.image()
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / 'toy.nii.gz')
            sitk.WriteImage(image, path)
            read = sitk.ReadImage(path, sitk.sitkFloat32)
            self.assertEqual(image.GetSize(), read.GetSize())
            for attr in ('Spacing', 'Origin', 'Direction'):
                np.testing.assert_allclose(getattr(image, 'Get' + attr)(),
                                           getattr(read, 'Get' + attr)(), atol=1e-5, rtol=0)
            np.testing.assert_array_equal(sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(read))

    def test_real_b_spline_resampling_geometry(self):
        image = self.image()
        result = upstream.resample_img(image, (1.4, 2.2, 6.))
        self.assertEqual(result.GetSize(), (20, 18, 10))
        self.assertEqual(result.GetOrigin(), image.GetOrigin())
        self.assertEqual(result.GetDirection(), image.GetDirection())
        self.assertEqual(result.GetSpacing(), (1.4, 2.2, 6.))
        np.testing.assert_allclose(sitk.GetArrayFromImage(result), 35, atol=1e-4)

    def test_physical_crop_rotated_direction_and_anisotropic_spacing(self):
        image = self.image()
        low = upstream.resample_img(image, (1.4, 2.2, 6.))
        mask = np.zeros((10, 18, 20), dtype=np.uint8)
        mask[2:6, 3:7, 4:9] = 1
        segmentation = sitk.GetImageFromArray(mask)
        segmentation.CopyInformation(low)
        cropped, coordinates = upstream.CropPancreasROI(image, segmentation, [1.4, 2.2, 3.])
        self.assertEqual(coordinates, dict(x_start=6, x_finish=18, y_start=4,
                                           y_finish=14, z_start=3, z_finish=11))
        self.assertEqual(cropped.GetSize(), (12, 10, 8))
        np.testing.assert_allclose(cropped.GetOrigin(), image.TransformIndexToPhysicalPoint((6, 4, 3)))

    def test_empty_localization_fails_not_negative(self):
        image = self.image()
        mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(image)
        with self.assertRaises(AssertionError):
            upstream.CropPancreasROI(image, mask, [100, 50, 15])

    def test_actual_masking_matches_guard_without_mutating_input(self):
        segmentation = np.zeros((9, 10, 11), dtype=np.uint8)
        segmentation[4, 4, 4] = 4
        segmentation[4, 7, 7] = 5
        segmentation[7, 7, 7] = 2
        probability = np.full(segmentation.shape, .4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / 'segmentation.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(segmentation), path)
            actual = upstream.PostProcessing({'probabilities': np.stack([1-probability, probability])}, path)
        expected, _ = postprocess_probability(probability, segmentation)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(probability, np.full(segmentation.shape, .4, dtype=np.float32))

    def test_candidate_ten_voxel_boundary(self):
        for count in (10, 11):
            prediction = np.zeros((4, 4, 16), dtype=np.float32)
            prediction[1, 1, :count] = .5
            candidates, confidences, _ = extract_lesion_candidates(prediction)
            self.assertEqual(len(confidences), 0 if count == 10 else 1)
            self.assertEqual(np.count_nonzero(candidates), 0 if count == 10 else 11)

    def test_remote_peak_changes_faint_candidate_without_changing_it(self):
        prediction = np.zeros((12, 12, 12), dtype=np.float32)
        prediction[1:3, 1:3, 1:4] = .15  # twelve connected voxels
        before, _, _ = extract_lesion_candidates(prediction)
        prediction[8:10, 8:10, 8:11] = .8
        after, _, _ = extract_lesion_candidates(prediction)
        self.assertEqual(np.count_nonzero(before[1:3, 1:3, 1:4]), 12)
        self.assertEqual(np.count_nonzero(after[1:3, 1:3, 1:4]), 0)

    def test_dynamic_fast_does_not_enforce_five_candidate_limit(self):
        prediction = np.zeros((5, 5, 40), dtype=np.float32)
        for start in range(0, 35, 5):
            prediction[1:3, 1:3, start:start+3] = .5
        _, confidences, _ = extract_lesion_candidates(prediction)
        self.assertEqual(len(confidences), 7)

    def test_actual_native_expansion_geometry(self):
        image = self.image()
        coordinates = dict(x_start=6, x_finish=18, y_start=4, y_finish=14,
                           z_start=3, z_finish=11)
        prediction = np.zeros((8, 10, 12), dtype=np.float32)
        prediction[2:4, 2:4, 2:5] = .6
        result, score = upstream.GetFullSizDetectionMap(prediction, coordinates, image)
        candidates, _, _ = extract_lesion_candidates(prediction)
        expected = expand_map(candidates, (20, 36, 40), coordinates)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(result), expected)
        self.assertEqual(score, float(np.max(candidates)))
        self.assertEqual(result.GetOrigin(), image.GetOrigin())
        self.assertEqual(result.GetSpacing(), image.GetSpacing())
        self.assertEqual(result.GetDirection(), image.GetDirection())


if __name__ == '__main__':
    unittest.main()
