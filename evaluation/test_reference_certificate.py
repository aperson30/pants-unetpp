import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.recover_geometry_sources import ARCHIVES, sha
from evaluation.verify_versioned_tumor_reference import verify


class ReferenceCertificateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'evaluation'; self.root.mkdir()
        self.reference = Path(self.temporary.name) / 'reference'; self.reference.mkdir()
        self.cid = 'PanTS_00009001'; self.cases = {self.cid}
        for folder in (self.root / 'test_ground_truth', self.reference / 'test_ground_truth'):
            folder.mkdir()
        self.original = self.root / 'test_ground_truth' / f'{self.cid}.nii.gz'
        self.derived = self.reference / 'test_ground_truth' / f'{self.cid}.nii.gz'
        array = np.zeros((2, 3, 4), np.uint8); array[1, 1, 1] = 28; array[0, 0, 0] = 17
        wrong = np.eye(4); wrong[0, 3] = 200
        nib.save(nib.Nifti1Image(array, wrong), self.original)
        nib.save(nib.Nifti1Image((array == 28).astype(np.uint8) * 28, np.eye(4)), self.derived)
        (self.root / 'evaluation_manifest.json').write_text('{}')
        provenances = {}
        for tag in ('unetpp_ds', 'unetpp_nods'):
            folder = self.root / tag; folder.mkdir()
            path = folder / 'prediction_provenance.json'
            path.write_text(json.dumps({'tumor_class': 28, 'inputs': {self.cid: 'a' * 64}}))
            provenances[tag] = sha(path)
        self.row = {'shape': [2, 3, 4], 'tumor_voxels': 1, 'prediction_affine': np.eye(4).tolist(),
                    'original_gt_sha256': sha(self.original), 'prediction_ct_sha256': 'a' * 64,
                    'tumor_index_equality': True, 'source_geometry_matches_original_ct': True}
        self.report = {'cases': {self.cid: self.row},
                       'archives': {name: digest for name, _, digest, _ in ARCHIVES}}
        self.audit = Path(self.temporary.name) / 'audit.json'
        self.audit.write_text(json.dumps(self.report))
        self.certificate = {'schema': 'pants-tumor-reference-v1', 'cases': [self.cid],
                            'source_audit_sha256': sha(self.audit),
                            'original_evaluation_manifest_sha256': sha(self.root / 'evaluation_manifest.json'),
                            'prediction_provenance_sha256': provenances,
                            'original_gt_sha256': {self.cid: sha(self.original)},
                            'reference_sha256': {self.cid: sha(self.derived)}}
        self.publish()

    def publish(self):
        (self.reference / 'tumor_reference_manifest.json').write_text(json.dumps(self.certificate))

    def check(self):
        return verify(self.root, self.reference, self.audit, expected_cases=self.cases)

    def update_audit(self):
        self.audit.write_text(json.dumps(self.report))
        self.certificate['source_audit_sha256'] = sha(self.audit); self.publish()

    def test_valid_and_read_only(self):
        before = {str(p): sha(p) for p in Path(self.temporary.name).rglob('*') if p.is_file()}
        self.assertEqual(self.check()['cases'], [self.cid])
        self.assertEqual(before, {str(p): sha(p) for p in Path(self.temporary.name).rglob('*') if p.is_file()})

    def test_fixture_is_not_production_cohort(self):
        with self.assertRaisesRegex(RuntimeError, 'cohort'):
            verify(self.root, self.reference, self.audit)

    def test_changed_reference_hash(self):
        with self.derived.open('ab') as output: output.write(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'reference identity'):
            self.check()

    def test_changed_tumor_even_with_updated_hash(self):
        nib.save(nib.Nifti1Image(np.zeros((2, 3, 4), np.uint8), np.eye(4)), self.derived)
        self.certificate['reference_sha256'][self.cid] = sha(self.derived); self.publish()
        with self.assertRaisesRegex(RuntimeError, 'tumor voxels'):
            self.check()

    def test_changed_geometry_even_with_updated_hash(self):
        image = nib.load(self.derived); data = np.asanyarray(image.dataobj)
        wrong = np.eye(4); wrong[0, 3] = 5
        nib.save(nib.Nifti1Image(data, wrong), self.derived)
        self.certificate['reference_sha256'][self.cid] = sha(self.derived); self.publish()
        with self.assertRaisesRegex(RuntimeError, 'geometry'):
            self.check()

    def test_missing_source_gate(self):
        self.row['tumor_index_equality'] = False; self.update_audit()
        with self.assertRaisesRegex(RuntimeError, 'complete certification'):
            self.check()

    def test_wrong_ct_identity(self):
        self.row['prediction_ct_sha256'] = 'b' * 64; self.update_audit()
        with self.assertRaisesRegex(RuntimeError, 'CT identity'):
            self.check()

    def test_wrong_archive(self):
        self.report['archives'][ARCHIVES[0][0]] = 'b' * 64; self.update_audit()
        with self.assertRaisesRegex(RuntimeError, 'archive identities'):
            self.check()

    def test_changed_original(self):
        with self.original.open('ab') as output: output.write(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'original GT identity'):
            self.check()


if __name__ == '__main__':
    unittest.main()
