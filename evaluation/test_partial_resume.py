import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.audit_partial_resume import audit_partial
from evaluation.predict_and_shrink import file_hash, write_scores
from evaluation.test_artifact_audit import ArtifactAuditTest


class PartialResumeTests(unittest.TestCase):
    def fixture(self, root):
        evaluation = root / 'evaluation'; evaluation.mkdir()
        frozen = root / 'frozen'; (frozen / 'evaluation').mkdir(parents=True)
        source = frozen / 'evaluation/predict_and_shrink.py'
        source.write_text('test fixture, not executable source')
        (frozen / '.frozen_commit').write_text('revision')
        ArtifactAuditTest().fixture(evaluation)
        manifest = {'training_commit': 'training', 'first_evaluation_commit': 'revision',
                    'source_hashes': {'evaluation/predict_and_shrink.py': file_hash(source)}}
        (evaluation / 'evaluation_manifest.json').write_text(json.dumps(manifest))
        for tag in ('unetpp_ds', 'unetpp_nods', 'default_ds', 'default_nods'):
            path = evaluation / tag / 'prediction_provenance.json'
            value = json.loads(path.read_text()); value['predictor_sha256'] = file_hash(source)
            path.write_text(json.dumps(value))
        return evaluation, frozen

    def test_complete_pair_counts_without_declaring_grid_complete(self):
        with tempfile.TemporaryDirectory() as name:
            evaluation, frozen = self.fixture(Path(name))
            result = audit_partial(evaluation, frozen, 2)
            self.assertEqual(result['cells']['unetpp_ds']['saved_pairs'], 2)
            self.assertIn('NOT complete', result['scope'])

    def test_partial_pair_is_pending_and_not_modified(self):
        with tempfile.TemporaryDirectory() as name:
            evaluation, frozen = self.fixture(Path(name))
            out = evaluation / 'unetpp_ds'
            write_scores(out / 'max_tumor_probs.csv', {'PanTS_00009001': .8})
            mask = out / 'PanTS_00009002.nii.gz'; before = file_hash(mask)
            cell = audit_partial(evaluation, frozen, 2)['cells']['unetpp_ds']
            self.assertEqual(cell['remaining'], 1)
            self.assertEqual(cell['mask_without_score'], ['PanTS_00009002'])
            self.assertEqual(file_hash(mask), before)

    def test_changed_source_bad_geometry_wrong_checkpoint_extra_case_rejected(self):
        for failure in ('source', 'geometry', 'checkpoint', 'extra'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as name:
                evaluation, frozen = self.fixture(Path(name))
                if failure == 'source':
                    (frozen / 'evaluation/predict_and_shrink.py').write_text('changed')
                elif failure == 'checkpoint':
                    path = evaluation / 'unetpp_ds/prediction_provenance.json'
                    value = json.loads(path.read_text()); value['checkpoint_sha256'] = 'changed'
                    path.write_text(json.dumps(value))
                elif failure == 'extra':
                    write_scores(evaluation / 'unetpp_ds/max_tumor_probs.csv', {'unexpected': .8})
                else:
                    affine = np.eye(4); affine[0, 3] = 10
                    nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), np.uint8), affine),
                             evaluation / 'unetpp_ds/PanTS_00009001.nii.gz')
                with self.assertRaises(RuntimeError):
                    audit_partial(evaluation, frozen, 2)


if __name__ == '__main__':
    unittest.main()
