import copy
import unittest
from data_gate_report import report


class DataGateReportTests(unittest.TestCase):
    def fixtures(self):
        strict = {"revision": "rev", "metadata_sha256": "hash"}
        expanded = strict | {"candidate_groups": [{"candidate_group": "g", "candidate_patient": "p",
            "noncontrast": [{"case_id": "nc", "phase_hint": "non-contrast"}],
            "contrast": [{"case_id": "ce", "phase_hint": "venous"}]}]}
        prior = {"metadata_sha256": "hash", "archive_scan_complete": True,
                 "requested_targets_complete": True,
                 "inspected_candidate_pancreatic_masks": {"nc": {"foreground_voxels": 10}}}
        extension = strict | {"masks": {}}
        return strict, expanded, prior, extension

    def test_unknown_is_not_absent_or_verified(self):
        result = report(*self.fixtures())
        self.assertEqual(result["unknown_mask_cases"], ["ce"])
        self.assertEqual(result["verified_tumor_training_pairs"], 0)
        self.assertFalse(result["frozen_training_manifest"])

    def test_nonempty_both_does_not_certify_pair(self):
        fixtures = self.fixtures()
        fixtures[3]["masks"]["ce"] = {"foreground_voxels": 100}
        result = report(*fixtures)
        self.assertEqual(result["candidate_patients_with_nonempty_both_phase_masks"], 1)
        self.assertFalse(result["groups"][0]["training_eligible"])

    def test_mixed_incomplete_or_overwriting_evidence_rejected(self):
        for mutate in (lambda x: x[2].update(archive_scan_complete=False),
                       lambda x: x[3].update(metadata_sha256="other"),
                       lambda x: x[3]["masks"].update(nc={"foreground_voxels": 0})):
            fixtures = copy.deepcopy(self.fixtures())
            mutate(fixtures)
            with self.assertRaises(ValueError):
                report(*fixtures)
