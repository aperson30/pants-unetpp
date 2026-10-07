import json
import shutil
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.audit_test_artifacts import TAGS
from evaluation.audit_versioned_predictions import PROTOCOL, audit
from evaluation.recover_geometry_sources import ARCHIVES, sha
from evaluation import test_artifact_audit as artifact_fixtures


class VersionedPredictionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / 'evaluation'; self.root.mkdir()
        artifact_fixtures.ArtifactAuditTest().fixture(self.root)
        self.reference = Path(self.temporary.name) / 'reference'; self.reference.mkdir()
        (self.reference / 'test_ground_truth').mkdir()
        self.cases = {'PanTS_00009001', 'PanTS_00009002'}
        (self.root / 'evaluation_manifest.json').write_text(json.dumps({
            'source_hashes': {'evaluation/predict_and_shrink.py': '3' * 64}}))
        for tag in TAGS:
            path = self.root / tag / 'prediction_provenance.json'
            value = json.loads(path.read_text())
            value.update(predictor_sha256='3' * 64, plans_sha256='1' * 64,
                         dataset_sha256='2' * 64, protocol=PROTOCOL)
            path.write_text(json.dumps(value))
        rows = {}; original_hashes = {}; reference_hashes = {}
        for cid in sorted(self.cases):
            original = self.root / 'test_ground_truth' / f'{cid}.nii.gz'
            data = np.asanyarray(nib.load(original).dataobj)
            wrong = np.eye(4); wrong[0, 3] = 200
            nib.save(nib.Nifti1Image(data, wrong), original)
            derived = self.reference / 'test_ground_truth' / original.name
            nib.save(nib.Nifti1Image((data == 28).astype(np.uint8) * 28, np.eye(4)), derived)
            rows[cid] = {'prediction_ct_sha256': 'ct1' if cid.endswith('1') else 'ct2',
                         'original_gt_sha256': sha(original), 'shape': list(data.shape),
                         'tumor_voxels': int((data == 28).sum()), 'prediction_affine': np.eye(4).tolist(),
                         'tumor_index_equality': True, 'source_geometry_matches_original_ct': True}
            original_hashes[cid] = sha(original); reference_hashes[cid] = sha(derived)
        self.source = Path(self.temporary.name) / 'source_audit.json'
        self.source.write_text(json.dumps({'cases': rows,
            'archives': {name: digest for name, _, digest, _ in ARCHIVES}}))
        certificate = {'schema': 'pants-tumor-reference-v1', 'cases': sorted(self.cases),
                       'source_audit_sha256': sha(self.source),
                       'original_evaluation_manifest_sha256': sha(self.root / 'evaluation_manifest.json'),
                       'prediction_provenance_sha256': {tag: sha(self.root / tag / 'prediction_provenance.json')
                                                        for tag in ('unetpp_ds', 'unetpp_nods')},
                       'original_gt_sha256': original_hashes, 'reference_sha256': reference_hashes}
        (self.reference / 'tumor_reference_manifest.json').write_text(json.dumps(certificate))

    def check(self):
        return audit(self.root, self.reference, self.source, expected_cases=self.cases)

    def test_certified_ct_geometry_not_bad_original_header(self):
        before = {str(p): sha(p) for p in Path(self.temporary.name).rglob('*') if p.is_file()}
        result = self.check()
        self.assertEqual(result['cases_per_cell'], 2)
        self.assertEqual(result['gt_positive_cases'], 1)
        self.assertEqual(result['gt_negative_cases'], 1)
        self.assertEqual(before, {str(p): sha(p) for p in Path(self.temporary.name).rglob('*') if p.is_file()})

    def test_partial_resume_is_not_a_final_scoring_pass(self):
        shutil.rmtree(self.root / 'default_ds')
        shutil.rmtree(self.root / 'default_nods')
        path = self.root / 'unetpp_nods/PanTS_00009002.nii.gz'; path.unlink()
        (self.root / 'unetpp_nods/max_tumor_probs.csv').write_text(
            'case_id,max_tumor_probability\nPanTS_00009001,0.8\n')
        before = {str(p): sha(p) for p in Path(self.temporary.name).rglob('*') if p.is_file()}
        result = audit(self.root, self.reference, self.source, expected_cases=self.cases, partial=True)
        self.assertFalse(result['final_scoring_ready'])
        self.assertEqual(result['schema'], 'pants-partial-prediction-resume-audit-v1')
        self.assertEqual(len(result['completed_cases']['unetpp_ds']), 2)
        self.assertEqual(result['remaining_cases']['unetpp_nods'], ['PanTS_00009002'])
        self.assertEqual(result['remaining_cases']['default_ds'], sorted(self.cases))
        self.assertEqual(before, {str(p): sha(p) for p in Path(self.temporary.name).rglob('*') if p.is_file()})
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            self.check()

    def test_partial_resume_rejects_orphans_and_unprovenanced_outputs(self):
        path = self.root / 'default_ds/PanTS_00009001.nii.gz'; path.unlink()
        with self.assertRaisesRegex(RuntimeError, 'mask or probability'):
            audit(self.root, self.reference, self.source, expected_cases=self.cases, partial=True)
        (self.root / 'default_ds/prediction_provenance.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'unprovenanced'):
            audit(self.root, self.reference, self.source, expected_cases=self.cases, partial=True)

    def test_missing_mask_or_score_or_extra_case(self):
        for failure in ('mask', 'score', 'extra'):
            with self.subTest(failure=failure):
                # Each mutation is reverted without changing any certificate.
                path = self.root / 'default_ds/PanTS_00009001.nii.gz'
                if failure == 'mask':
                    renamed = path.with_suffix('.hidden'); path.rename(renamed)
                    try:
                        with self.assertRaisesRegex(RuntimeError, 'mask or probability'): self.check()
                    finally: renamed.rename(path)
                elif failure == 'score':
                    score = self.root / 'default_ds/max_tumor_probs.csv'; original = score.read_bytes()
                    score.write_text('case_id,max_tumor_probability\nPanTS_00009001,0.8\n')
                    try:
                        with self.assertRaisesRegex(RuntimeError, 'mask or probability'): self.check()
                    finally: score.write_bytes(original)
                else:
                    extra = self.root / 'default_ds/PanTS_00009999.nii.gz'; extra.write_bytes(path.read_bytes())
                    try:
                        with self.assertRaisesRegex(RuntimeError, 'mask or probability'): self.check()
                    finally: extra.unlink()

    def test_wrong_checkpoint_input_predictor_protocol_and_plan(self):
        path = self.root / 'default_ds/prediction_provenance.json'; original = path.read_bytes()
        for field in ('checkpoint_sha256', 'inputs', 'predictor_sha256', 'protocol', 'plans_sha256'):
            with self.subTest(field=field):
                value = json.loads(original)
                if field == 'inputs': value[field]['PanTS_00009001'] = 'wrong'
                else: value[field] = '4' * 64
                path.write_text(json.dumps(value))
                try:
                    with self.assertRaises(RuntimeError): self.check()
                finally: path.write_bytes(original)

    def test_wrong_prediction_geometry_and_label(self):
        path = self.root / 'default_ds/PanTS_00009001.nii.gz'; original = path.read_bytes()
        for failure in ('geometry', 'label'):
            with self.subTest(failure=failure):
                data = np.zeros((3, 4, 5), np.uint8); affine = np.eye(4)
                if failure == 'geometry': affine[0, 3] = 200
                else: data[0, 0, 0] = 17
                nib.save(nib.Nifti1Image(data, affine), path)
                try:
                    with self.assertRaises(RuntimeError): self.check()
                finally: path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
