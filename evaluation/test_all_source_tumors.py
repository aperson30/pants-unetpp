import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from evaluation.audit_all_source_tumors import check_tumor, selected_members, check_ct_identity, publish_report


class AllSourceAuditTests(unittest.TestCase):
    def test_default_identity_gate_stays_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'exact frozen'):
            check_ct_identity('case', 'actual', 'expected', False)
        self.assertFalse(check_ct_identity('case', 'actual', 'expected', True))
        self.assertTrue(check_ct_identity('case', 'same', 'same', False))

    def test_diagnostic_never_publishes_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            publish_report(folder, {'ct_identity_mismatches': []}, True)
            self.assertTrue((folder / 'source_audit_diagnostic.json').is_file())
            self.assertFalse((folder / 'all_901_source_audit.json').exists())

    def test_target_equality_accepts_wrong_combined_header_not_wrong_voxels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array = np.zeros((3, 4, 5), np.uint8); array[1, 2, 3] = 1
            affine = np.eye(4); wrong = affine.copy(); wrong[0, 3] = 200
            source = root / 'source.nii.gz'; gt = root / 'gt.nii.gz'
            nib.save(nib.Nifti1Image(array, affine), source)
            nib.save(nib.Nifti1Image(array * 28, wrong), gt)
            ct = {'shape': list(array.shape), 'raw_affine': affine.tolist(), 'prediction_affine': affine.tolist()}
            result = check_tumor(source, gt, ct)
            self.assertFalse(result['saved_gt_matches_prediction_grid'])
            nib.save(nib.Nifti1Image(np.flip(array, axis=1) * 28, wrong), gt)
            with self.assertRaisesRegex(RuntimeError, 'voxels differ'):
                check_tumor(source, gt, ct)
            nib.save(nib.Nifti1Image(array, wrong), source)
            with self.assertRaisesRegex(RuntimeError, 'geometry differs'):
                check_tumor(source, gt, ct)

    def test_missing_duplicate_and_unsafe_members_rejected(self):
        cid = 'PanTS_00009001'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for paths, message in [([], 'missing'),
                                   ([f'{cid}/ct.nii.gz'] * 2, 'duplicate'),
                                   ([f'../{cid}/ct.nii.gz'], 'unsafe')]:
                archive = root / 'test.tar.gz'
                with tarfile.open(archive, 'w:gz') as output:
                    for name in paths:
                        member = tarfile.TarInfo(name); member.size = 3
                        output.addfile(member, io.BytesIO(b'abc'))
                with patch('evaluation.audit_all_source_tumors.CASES', {cid}):
                    with self.assertRaisesRegex(RuntimeError, message):
                        list(selected_members(archive, 'ct.nii.gz', root))


if __name__ == '__main__':
    unittest.main()
