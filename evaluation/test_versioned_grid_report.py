import json
import unittest

from evaluation.test_versioned_scoring import VersionedScoringTests
from evaluation.verify_versioned_grid_report import verify
from evaluation.recover_geometry_sources import sha


class FinalReportTests(unittest.TestCase):
    def test_independent_report_and_tamper_refusals(self):
        VersionedScoringTests.setUp(self)
        VersionedScoringTests.run_score(self)
        def check():
            return verify(self.root, self.reference, self.source, self.metric_script,
                          self.destination, expected_cases=self.cases)
        before = {str(p): sha(p) for p in self.root.parent.rglob('*') if p.is_file()}
        self.assertEqual(check()['cases_per_cell'], 2)
        self.assertEqual(before, {str(p): sha(p) for p in self.root.parent.rglob('*') if p.is_file()})
        report_path = self.destination / 'grid_metrics.json'
        original_report = report_path.read_bytes()
        report = json.loads(original_report)
        metrics_path = self.destination / 'default_ds/metrics.json'
        original_metrics = metrics_path.read_bytes()
        # Even consistently edited JSON + hashes must agree with independent CSV math.
        altered = json.loads(original_metrics); altered['AUC'] = 0.5
        metrics_path.write_text(json.dumps(altered))
        report['metrics']['default_ds'] = altered
        report['metric_artifact_sha256']['default_ds/metrics.json'] = sha(metrics_path)
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(RuntimeError, 'independent CSV aggregation'):
            check()
        metrics_path.write_bytes(original_metrics); report_path.write_bytes(original_report)
        csv_path = self.destination / 'default_ds/per_case.csv'; original_csv = csv_path.read_bytes()
        csv_path.write_bytes(original_csv + original_csv.splitlines(keepends=True)[1])
        report = json.loads(original_report)
        report['metric_artifact_sha256']['default_ds/per_case.csv'] = sha(csv_path)
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(RuntimeError, 'duplicate cases'):
            check()
        csv_path.write_bytes(original_csv); report_path.write_bytes(original_report)
        report = json.loads(original_report); report['schema'] = 'pants-partial-prediction-resume-audit-v1'
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(RuntimeError, 'unsupported'):
            check()


if __name__ == '__main__':
    unittest.main()
