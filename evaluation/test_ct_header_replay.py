import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.diagnose_ct_header_replay import diagnose
from evaluation.recover_geometry_sources import sha


class HeaderReplayTests(unittest.TestCase):
    def test_preserves_voxels_and_rejects_spatially_different_reference(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); raw = root / 'raw.nii.gz'; fixed = root / 'fixed.nii.gz'; gt = root / 'gt.nii.gz'
            array = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
            for path in (raw, fixed):
                nib.save(nib.Nifti1Image(array, np.eye(4)), path)
            nib.save(nib.Nifti1Image(np.zeros(array.shape, np.uint8), np.eye(4)), gt)
            hashes = {path: sha(path) for path in (raw, fixed, gt)}
            diagnose(raw, fixed, gt, root / 'out', sha(fixed))
            self.assertEqual(hashes, {path: sha(path) for path in hashes})
            wrong = np.eye(4); wrong[0, 3] = 100
            nib.save(nib.Nifti1Image(np.zeros(array.shape, np.uint8), wrong), gt)
            with self.assertRaisesRegex(RuntimeError, 'not the same'):
                diagnose(raw, fixed, gt, root / 'refused', sha(fixed))


if __name__ == '__main__':
    unittest.main()
