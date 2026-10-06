import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.build_versioned_tumor_reference import write_reference, build
from evaluation.recover_geometry_sources import sha


class VersionedReferenceTests(unittest.TestCase):
    def test_builder_refuses_diagnostic_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); audit = root / 'diagnostic.json'
            audit.write_text('{"diagnostic_only": true}')
            destination = root / 'refused'
            with self.assertRaisesRegex(RuntimeError, 'not a source certificate'):
                build(root, audit, sha(audit), destination)
            self.assertFalse(destination.exists())

    def test_preserves_tumor_and_original_bytes_but_uses_ct_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); original = root / 'original.nii.gz'; output = root / 'derived.nii.gz'
            array = np.zeros((3, 4, 5), np.uint8); array[1, 2, 3] = 28; array[0, 0, 0] = 17
            wrong = np.eye(4); wrong[0, 3] = 200
            nib.save(nib.Nifti1Image(array, wrong), original)
            original_hash = sha(original)
            row = {'original_gt_sha256': original_hash, 'shape': list(array.shape),
                   'tumor_voxels': 1, 'prediction_affine': np.eye(4).tolist()}
            write_reference(original, output, row)
            self.assertEqual(sha(original), original_hash)
            image = nib.load(output)
            self.assertTrue(np.array_equal(np.asanyarray(image.dataobj) == 28, array == 28))
            self.assertEqual(np.asanyarray(image.dataobj)[0, 0, 0], 0)
            self.assertTrue(np.array_equal(image.affine, np.eye(4)))
            row['original_gt_sha256'] = '0' * 64
            with self.assertRaisesRegex(RuntimeError, 'changed after'):
                write_reference(original, root / 'refused.nii.gz', row)
            self.assertFalse((root / 'refused.nii.gz').exists())


if __name__ == '__main__':
    unittest.main()
