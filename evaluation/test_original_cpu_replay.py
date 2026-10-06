import shutil
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from data_conversion import fix_affine_orthonormality as correction
from evaluation.recover_geometry_sources import sha
from evaluation.replay_ct_original_cpu import replay


class OriginalCpuReplayTests(unittest.TestCase):
    def test_exact_replay_and_original_bytes_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = root / 'PanTS_00009812_0000.nii.gz'
            affine = np.eye(4); affine[0, 1] = 0.001
            data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
            nib.save(nib.Nifti1Image(data, affine), raw)
            code = root / 'frozen.py'; shutil.copyfile(correction.__file__, code)
            raw_hash = sha(raw); code_hash = sha(code)
            baseline = root / 'baseline'; baseline.mkdir()
            shutil.copyfile(raw, baseline / raw.name); correction.fix_folder(baseline)
            expected = sha(baseline / raw.name)
            result = replay(raw, root / 'replay', code, expected, raw_hash, code_hash)
            self.assertTrue(result['exact_prediction_input_match'])
            self.assertTrue(result['diagnostic_only'])
            self.assertTrue(result['source_voxels_unchanged'])
            self.assertEqual(result['corrected_count'], 1)
            self.assertEqual(sha(raw), raw_hash); self.assertEqual(sha(code), code_hash)
            self.assertFalse((root / '__pycache__').exists())

    def test_wrong_original_identity_refuses_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = root / 'raw.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((2, 3, 4), np.uint8), np.eye(4)), raw)
            code = Path(correction.__file__); destination = root / 'refused'
            with self.assertRaisesRegex(RuntimeError, 'identity differs'):
                replay(raw, destination, code, '0' * 64, '0' * 64, sha(code))
            self.assertFalse(destination.exists())

    def test_wrong_expected_hash_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = root / 'raw.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((2, 3, 4), np.uint8), np.eye(4)), raw)
            code = Path(correction.__file__)
            result = replay(raw, root / 'diagnostic', code, '0' * 64, sha(raw), sha(code))
            self.assertFalse(result['exact_prediction_input_match'])
            self.assertTrue(result['diagnostic_only'])


if __name__ == '__main__':
    unittest.main()
