import json
import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np
from evaluation.audit_test_artifacts import TAGS, audit
from evaluation.predict_and_shrink import write_scores


class ArtifactAuditTest(unittest.TestCase):
    def fixture(self, root):
        (root / 'test_ground_truth').mkdir()
        (root / 'evaluation_manifest.json').write_text('{}')
        checkpoints = {key: {'checkpoint_sha256': tag} for tag, key in TAGS.items()}
        (root / 'checkpoint_validation_provenance.json').write_text(json.dumps(checkpoints))
        inputs = {'PanTS_00009001': 'ct1', 'PanTS_00009002': 'ct2'}
        for tag in TAGS:
            (root / tag).mkdir()
            (root / tag / 'prediction_provenance.json').write_text(json.dumps({
                'checkpoint_sha256': tag, 'tumor_class': 28, 'inputs': inputs}))
            write_scores(root / tag / 'max_tumor_probs.csv', {'PanTS_00009001': .8, 'PanTS_00009002': .2})
        for index, cid in enumerate(inputs):
            data = np.zeros((3, 4, 5), dtype=np.uint8)
            if index == 0:
                data[0, 0, 0] = 28
            for folder in ['test_ground_truth', *TAGS]:
                nib.save(nib.Nifti1Image(data, np.eye(4)), root / folder / f'{cid}.nii.gz')

    def test_pass_and_bit_change_changes_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.fixture(root)
            before = audit(root, 2)
            self.assertEqual(before['gt_positive_cases'], 1)
            self.assertEqual(len(before['artifact_sha256']), 18)
            path = root / 'default_ds/PanTS_00009001.nii.gz'
            data = np.zeros((3, 4, 5), dtype=np.uint8)
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)
            self.assertNotEqual(audit(root, 2), before)

    def test_reject_bad_geometry_labels_missing_case_and_score(self):
        for failure in ('geometry', 'label', 'missing', 'score', 'checkpoint', 'input'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); self.fixture(root)
                out = root / 'default_ds'
                path = out / 'PanTS_00009001.nii.gz'
                if failure == 'missing':
                    path.unlink()
                elif failure == 'score':
                    write_scores(out / 'max_tumor_probs.csv', {'PanTS_00009001': .8})
                elif failure in ('checkpoint', 'input'):
                    p = out / 'prediction_provenance.json'
                    value = json.loads(p.read_text())
                    if failure == 'checkpoint': value['checkpoint_sha256'] = 'wrong'
                    else: value['inputs']['PanTS_00009001'] = 'wrong'
                    p.write_text(json.dumps(value))
                else:
                    data = np.zeros((3, 4, 5), dtype=np.uint8)
                    affine = np.eye(4)
                    if failure == 'label': data[0, 0, 0] = 17
                    else: affine[0, 3] = 1
                    nib.save(nib.Nifti1Image(data, affine), path)
                with self.assertRaises(RuntimeError): audit(root, 2)


if __name__ == '__main__':
    unittest.main()
