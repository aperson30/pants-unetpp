import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.diagnose_corrected_ct_replay import compare_replays
from evaluation.recover_geometry_sources import sha


class CorrectedReplayTests(unittest.TestCase):
    def test_actual_function_and_manual_replay_agree_on_skewed_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'case_0000.nii.gz'
            values = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
            affine = np.eye(4); affine[0, 1] = 0.0002
            nib.save(nib.Nifti1Image(values, affine), source)
            original = sha(source)
            result = compare_replays(source, root, '0' * 64)
            self.assertEqual(result['frozen_files_changed'], 1)
            self.assertEqual(result['frozen_in_place_sha256'], result['manual_sha256'])
            self.assertTrue(result['frozen_preserves_source_values'])
            self.assertTrue(result['manual_preserves_source_values'])
            self.assertEqual(sha(source), original)
            self.assertFalse(result['frozen_matches_prediction_bytes'])


if __name__ == '__main__':
    unittest.main()
