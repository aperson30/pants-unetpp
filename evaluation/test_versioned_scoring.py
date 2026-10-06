import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import compute_tumor_metrics
from evaluation import test_versioned_predictions as prediction_fixtures
from evaluation.recover_geometry_sources import sha
from evaluation.score_versioned_grid_cpu import score
from evaluation.audit_versioned_predictions import audit


class VersionedScoringTests(unittest.TestCase):
    def setUp(self):
        prediction_fixtures.VersionedPredictionTests.setUp(self)
        self.metric_script = Path(compute_tumor_metrics.__file__)
        manifest_path = self.root / 'evaluation_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['source_hashes']['evaluation/compute_tumor_metrics.py'] = sha(self.metric_script)
        manifest_path.write_text(json.dumps(manifest))
        certificate_path = self.reference / 'tumor_reference_manifest.json'
        certificate = json.loads(certificate_path.read_text())
        certificate['original_evaluation_manifest_sha256'] = sha(manifest_path)
        certificate_path.write_text(json.dumps(certificate))
        self.destination = Path(self.temporary.name) / 'scores'

    def run_score(self):
        return score(self.root, self.reference, self.source, self.metric_script,
                     self.destination, expected_cases=self.cases, cell_timeout=60)

    def test_known_metrics_and_read_only_inputs(self):
        inputs = {str(p): sha(p) for p in Path(self.temporary.name).rglob('*') if p.is_file()}
        result = self.run_score()
        self.assertTrue((self.destination / 'grid_metrics.json').is_file())
        self.assertEqual(len(result['metrics']), 4)
        for metrics in result['metrics'].values():
            for name in ('DSC_tumor_mean', 'P_Sen', 'T_Sen', 'Spe', 'AUC'):
                self.assertEqual(metrics[name], 1)
            self.assertEqual(metrics['n_cases_evaluated'], 2)
        self.assertEqual(inputs, {path: sha(Path(path)) for path in inputs})

    def test_failed_worker_never_publishes_summary(self):
        with patch('evaluation.score_versioned_grid_cpu.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'fixture')):
            with self.assertRaises(subprocess.CalledProcessError): self.run_score()
        self.assertFalse((self.destination / 'grid_metrics.json').exists())

    def test_metric_identity_and_protected_destination_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'separate'):
            score(self.root, self.reference, self.source, self.metric_script, self.root / 'new_scores',
                  expected_cases=self.cases)
        manifest_path = self.root / 'evaluation_manifest.json'
        value = json.loads(manifest_path.read_text())
        value['source_hashes']['evaluation/compute_tumor_metrics.py'] = '0' * 64
        manifest_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, 'metric script differs'): self.run_score()
        self.assertFalse(self.destination.exists())

    def test_diagnostic_source_refused_before_outputs(self):
        report = json.loads(self.source.read_text()); report['diagnostic_only'] = True
        self.source.write_text(json.dumps(report))
        path = self.reference / 'tumor_reference_manifest.json'
        certificate = json.loads(path.read_text()); certificate['source_audit_sha256'] = sha(self.source)
        path.write_text(json.dumps(certificate))
        with self.assertRaisesRegex(RuntimeError, 'not a source certificate'): self.run_score()
        self.assertFalse(self.destination.exists())

    def test_changed_inputs_never_publish_summary(self):
        before = audit(self.root, self.reference, self.source, expected_cases=self.cases)
        after = dict(before, changed=True)
        def fake_worker(command, **kwargs):
            metrics = {'protocol': {'tumor_class': 28, 'T_Sen_connectivity': '6-neighbor'},
                       'n_cases_evaluated': 2, 'AUC_n_cases': 2,
                       'DSC_tumor_n_positive_cases': 1, 'P_Sen_n_positive_cases': 1,
                       'Spe_n_negative_cases': 1, 'T_Sen_n_true_tumors': 1,
                       **{name: 1.0 for name in ('DSC_tumor_mean', 'P_Sen', 'T_Sen', 'Spe', 'AUC')}}
            Path(command[command.index('--out-json') + 1]).write_text(json.dumps(metrics))
            Path(command[command.index('--per-case-csv') + 1]).write_text('fixture\n')
            return subprocess.CompletedProcess(command, 0)
        with patch('evaluation.score_versioned_grid_cpu.audit', side_effect=[before, after]), \
             patch('evaluation.score_versioned_grid_cpu.subprocess.run', side_effect=fake_worker):
            with self.assertRaisesRegex(RuntimeError, 'inputs changed'): self.run_score()
        self.assertFalse((self.destination / 'grid_metrics.json').exists())


if __name__ == '__main__':
    unittest.main()
