import unittest
from general_pair_split_proposal import propose


class SplitProposalTests(unittest.TestCase):
    def test_groups_of_same_patient_stay_together_and_are_not_certified(self):
        groups=[]
        for i,patient in enumerate(("a","b","a","c","d","e")):
            groups.append({"candidate_group":str(i),"candidate_patient":patient,
                           "noncontrast":[{"case_id":f"n{i}","phase_hint":"non-contrast"}],
                           "contrast":[{"case_id":f"v{i}","phase_hint":"venous"}]})
        r=propose({"groups":groups,"revision":"r","metadata_sha256":"h"})
        a=[row["provisional_split"] for row in r["rows"] if row["candidate_patient"] == "a"]
        self.assertEqual(len(set(a)),1)
        self.assertFalse(r["clinical_training_eligible"])
        self.assertTrue(all(not row["training_eligible"] for row in r["rows"]))
        self.assertEqual(r,propose({"groups":list(reversed(groups)),"revision":"r","metadata_sha256":"h"}))
