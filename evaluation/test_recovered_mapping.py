import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from evaluation.verify_recovered_tumor_mapping import compare_case


class SourceMappingTests(unittest.TestCase):
    def test_exact_target_lineage_with_bad_reference_header(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); shape = (3, 4, 5)
            ct_affine = np.eye(4); bad = np.eye(4); bad[0, 3] = 200
            blank = np.zeros(shape, np.uint8); tumor = blank.copy(); tumor[1, 2, 3] = 1
            for organ, data, affine in [('adrenal', blank, bad), ('pancreas', tumor, ct_affine),
                                        ('pancreatic_lesion', tumor, ct_affine)]:
                nib.save(nib.Nifti1Image(data, affine), root / f'{organ}.nii.gz')
            nib.save(nib.Nifti1Image(blank, ct_affine), root / 'ct.nii.gz')
            nib.save(nib.Nifti1Image(tumor * 28, bad), root / 'gt.nii.gz')
            with patch('evaluation.verify_recovered_tumor_mapping.CLASS_MAP',
                       {'adrenal': 1, 'pancreas': 17, 'pancreatic_lesion': 28}):
                result = compare_case(root, root / 'ct.nii.gz', root / 'gt.nii.gz')
                self.assertTrue(result['source_tumor_equals_saved_gt_tumor'])
                self.assertFalse(result['saved_gt_affine_matches_ct'])
                nib.save(nib.Nifti1Image(blank, bad), root / 'gt.nii.gz')
                with self.assertRaises(RuntimeError):
                    compare_case(root, root / 'ct.nii.gz', root / 'gt.nii.gz')


if __name__ == '__main__':
    unittest.main()
