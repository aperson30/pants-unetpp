import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from evaluation.audit_all_source_tumors import check_tumor, selected_members, check_ct_identity, publish_report, inspect_source_encoding
from evaluation.audit_all_source_tumors import verified_exact_replay
from evaluation.recover_geometry_sources import sha
from evaluation.replay_ct_original_cpu import replay
from data_conversion import fix_affine_orthonormality as correction


class AllSourceAuditTests(unittest.TestCase):
    def test_exact_replay_handoff_refuses_wrong_bytes_metadata_and_voxels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cid = 'PanTS_00009812'
            raw = root / f'{cid}_0000.nii.gz'
            affine = np.eye(4); affine[0, 1] = 0.001
            data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
            nib.save(nib.Nifti1Image(data, affine), raw)
            code = root / 'frozen.py'; shutil.copyfile(correction.__file__, code)
            baseline = root / 'baseline'; baseline.mkdir()
            shutil.copyfile(raw, baseline / raw.name); correction.fix_folder(baseline)
            expected = sha(baseline / raw.name)
            folder = root / 'replay'
            replay(raw, folder, code, expected, sha(raw), sha(code))
            original = {p: p.read_bytes() for p in (raw, code, folder / raw.name, folder / 'original_cpu_replay.json')}
            digest, result_affine, evidence = verified_exact_replay(folder, cid, raw, expected, code)
            self.assertEqual(digest, expected)
            self.assertTrue(evidence['exact_replay_source_voxels_verified'])
            np.testing.assert_array_equal(result_affine, nib.load(baseline / raw.name).affine)
            self.assertEqual(original, {p: p.read_bytes() for p in original})
            report_path = folder / 'original_cpu_replay.json'
            report = json.loads(report_path.read_text())
            for field, bad in [('raw_sha256', '0' * 64), ('correction_sha256', '0' * 64),
                               ('exact_prediction_input_match', False), ('affine', np.eye(4).tolist())]:
                altered = dict(report); altered[field] = bad
                report_path.write_text(json.dumps(altered))
                with self.assertRaises(RuntimeError):
                    verified_exact_replay(folder, cid, raw, expected, code)
            report_path.write_bytes(original[report_path])
            with self.assertRaises(RuntimeError):
                verified_exact_replay(folder, cid, raw, '0' * 64, code)
            # Even a self-consistent report/hash cannot authorize changed voxels.
            fixed = folder / raw.name
            nib.save(nib.Nifti1Image(data + 1, result_affine), fixed)
            changed = sha(fixed)
            report.update(replayed_sha256=changed, expected_prediction_sha256=changed,
                          affine=nib.load(fixed).affine.tolist())
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, 'geometry/voxels differ'):
                verified_exact_replay(folder, cid, raw, changed, code)

    def test_real_int8_scaling_keeps_exact_positive_voxel_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'scaled.nii.gz'; gt = root / 'gt.nii.gz'
            stored = np.full((2, 3, 4), -128, np.int8); stored[1, 1, 1] = 127
            image = nib.Nifti1Image(stored, np.eye(4))
            image.header.set_slope_inter(0.003921568859368563, 0.501960813999176)
            nib.save(image, source)
            decoded = np.asanyarray(nib.load(source).dataobj)
            self.assertEqual(decoded[0, 0, 0], 0)
            self.assertEqual(decoded[1, 1, 1], 1.0000000591389835)
            nib.save(nib.Nifti1Image((decoded > 0).astype(np.uint8) * 28, np.eye(4)), gt)
            before = source.read_bytes()
            row = {'shape': list(stored.shape), 'raw_affine': np.eye(4).tolist(), 'prediction_affine': np.eye(4).tolist()}
            self.assertTrue(check_tumor(source, gt, row)['tumor_index_equality'])
            self.assertEqual(source.read_bytes(), before)
            for bad in (-1e-8, 1e-8, 0.5, 2):
                data = np.zeros((2, 3, 4), np.float32); data[1, 1, 1] = bad
                nib.save(nib.Nifti1Image(data, np.eye(4)), source)
                with self.assertRaisesRegex(RuntimeError, 'not binary'):
                    check_tumor(source, gt, row)

    def test_encoding_inspection_is_not_a_source_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.nii.gz'; gt = root / 'gt.nii.gz'
            data = np.zeros((2, 3, 4), np.uint8); data[1, 1, 1] = 28
            nib.save(nib.Nifti1Image(data, np.eye(4)), source)
            nib.save(nib.Nifti1Image(data, np.eye(4)), gt)
            row = {'shape': list(data.shape), 'raw_affine': np.eye(4).tolist(), 'prediction_affine': np.eye(4).tolist()}
            with self.assertRaisesRegex(RuntimeError, 'not binary'):
                check_tumor(source, gt, row)
            evidence = inspect_source_encoding(source, gt)
            self.assertEqual(evidence['source_values_first_64'], [0, 28])
            self.assertTrue(evidence['positive_threshold_matches_saved_class28'])
            self.assertNotIn('tumor_index_equality', evidence)

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
