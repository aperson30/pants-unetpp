import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import nibabel as nib
import numpy as np
from evaluation.safe_artifacts import publish_ground_truth, retire_report
from evaluation.audit_test_artifacts import audit, verify_grid_report, TAGS
from evaluation.test_artifact_audit import ArtifactAuditTest
from evaluation.predict_and_shrink import invalidate_pending_scores, write_scores, read_scores


class SafeArtifactsTest(unittest.TestCase):
    def test_failed_scoring_and_audit_remove_current_success_markers(self):
        from evaluation.score_grid_cpu import main as score_main
        from evaluation.audit_test_artifacts import main as audit_main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / 'grid_metrics.json'
            report.write_text('{"old": true}')
            with patch.object(sys, 'argv', ['score', '--evaluation-dir', str(root)]), \
                 patch('evaluation.score_grid_cpu.audit', side_effect=RuntimeError('bad input')):
                with self.assertRaises(RuntimeError): score_main()
            self.assertFalse(report.exists())
            snapshot = root / 'artifact_audit.json'
            snapshot.write_text('{"old": true}')
            with patch.object(sys, 'argv', ['audit', '--evaluation-dir', str(root),
                                           '--out-manifest', str(snapshot)]), \
                 patch('evaluation.audit_test_artifacts.audit', side_effect=RuntimeError('bad input')):
                with self.assertRaises(RuntimeError): audit_main()
            self.assertFalse(snapshot.exists())

    def test_pending_invalidation_fails_before_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            csv = Path(temporary) / 'scores.csv'
            scores = {'case': .1, 'keep': .7}
            write_scores(csv, scores)
            with patch('evaluation.predict_and_shrink.write_scores', side_effect=OSError('disk')):
                with self.assertRaises(OSError):
                    invalidate_pending_scores(csv, scores, ['case'])
            self.assertEqual(read_scores(csv), scores)
            self.assertEqual(invalidate_pending_scores(csv, scores, ['case']), {'keep': .7})
            self.assertEqual(read_scores(csv), {'keep': .7})

    def test_atomic_gt_copy_and_partial_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'source'; dest = root / 'dest'
            source.mkdir(); dest.mkdir()
            name = 'PanTS_00009001.nii.gz'
            nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.uint8), np.eye(4)), source / name)
            (dest / name).write_bytes(b'interrupted')
            with patch('evaluation.safe_artifacts.shutil.copyfile', side_effect=OSError('disk')):
                with self.assertRaises(OSError): publish_ground_truth(source, dest)
            self.assertEqual((dest / name).read_bytes(), b'interrupted')
            publish_ground_truth(source, dest)
            self.assertEqual((dest / name).read_bytes(), (source / name).read_bytes())
            publish_ground_truth(source, dest)
            data = np.ones((3, 4, 5), dtype=np.uint8)
            nib.save(nib.Nifti1Image(data, np.eye(4)), source / name)
            with self.assertRaises(RuntimeError): publish_ground_truth(source, dest)

    def test_stale_report_is_rejected_and_preserved_as_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); ArtifactAuditTest().fixture(root)
            path = root / 'grid_metrics.json'
            metrics = {'n_cases_evaluated': 2,
                       'protocol': {'tumor_class': 28, 'T_Sen_connectivity': '6-neighbor'},
                       'DSC_tumor_mean': .5, 'P_Sen': 1., 'T_Sen': .5, 'Spe': 1., 'AUC': 1.}
            report = {'artifact_audit': audit(root, 2), 'project_protocol': True,
                      'metrics': {tag: metrics for tag in TAGS}}
            path.write_text(json.dumps(report))
            verify_grid_report(root, 2)
            for invalid in ({}, {**metrics, 'AUC': float('nan')}, {**metrics, 'n_cases_evaluated': 1}):
                report['metrics']['default_ds'] = invalid
                path.write_text(json.dumps(report))
                with self.assertRaises(RuntimeError): verify_grid_report(root, 2)
            report['metrics']['default_ds'] = metrics
            path.write_text(json.dumps(report))
            write_scores(root / 'default_ds/max_tumor_probs.csv',
                         {'PanTS_00009001': .3, 'PanTS_00009002': .2})
            with self.assertRaises(RuntimeError): verify_grid_report(root, 2)
            retire_report(path)
            self.assertFalse(path.exists())
            self.assertEqual(len(list(root.glob('grid_metrics.json.history.*'))), 1)


if __name__ == '__main__':
    unittest.main()
