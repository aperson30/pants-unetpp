import unittest

from paired_data_candidates import inventory


HEADER = "CancerVerse ID,Patient ID,Encrypted Accession Number,exam_date,phase,report\n"


class CandidateInventoryContracts(unittest.TestCase):
    def test_candidates_do_not_become_verified_and_no_raw_identity(self):
        raw = (HEADER + "CV_00000001,private-patient,private-accession,private-date,non-contrast,private-report\n"
               "CV_00000002,private-patient,private-accession,private-date,portal_venous,private-report\n").encode()
        result = inventory(raw)
        self.assertEqual(len(result["candidate_groups"]), 1)
        group = result["candidate_groups"][0]
        self.assertFalse(group["training_eligible"])
        self.assertFalse(group["registration_verified"])
        self.assertNotIn("private-", str(result))

    def test_no_pair_across_studies_or_unknown_phase(self):
        raw = (HEADER + "CV_00000001,p,a,date1,non-contrast,r\n"
               "CV_00000002,p,b,date2,portal_venous,r\n"
               "CV_00000003,p,a,date1,unknown,r\n").encode()
        self.assertEqual(inventory(raw)["candidate_groups"], [])

    def test_invalid_or_duplicate_case_rejected(self):
        for case in ("CV_00000001", "../../unsafe"):
            raw = (HEADER + "CV_00000001,p,a,d,non-contrast,r\n"
                   + f"{case},p,a,d,venous,r\n").encode()
            with self.assertRaises(ValueError):
                inventory(raw)

    def test_cross_accession_discovery_is_explicit_not_verification(self):
        raw = (HEADER + "CV_00000001,p,a,date1,non-contrast,r\n"
               "CV_00000002,p,b,date1,venous,r\n"
               "CV_00000003,p,c,date2,venous,r\n").encode()
        self.assertEqual(inventory(raw)["candidate_groups"], [])
        expanded = inventory(raw, "patient_date")
        self.assertEqual(len(expanded["candidate_groups"]), 1)
        self.assertFalse(expanded["candidate_groups"][0]["patient_linkage_verified"])
        self.assertEqual(len(expanded["candidate_groups"][0]["contrast"]), 1)
        with self.assertRaises(ValueError):
            inventory(raw, "patient_only")


if __name__ == "__main__":
    unittest.main()
