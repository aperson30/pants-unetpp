"""CPU-only patient-grouped venous candidate inventory, NOT training approval."""
import argparse
import hashlib
import json
from pathlib import Path


def propose(report):
    by_patient = {}
    for group in report["groups"]:
        patient = group["candidate_patient"]
        nc = sorted(row["case_id"] for row in group["noncontrast"] if row["phase_hint"] == "non-contrast")
        ce = sorted(row["case_id"] for row in group["contrast"] if row["phase_hint"] == "venous")
        if nc and ce:
            by_patient.setdefault(patient,[]).append({"candidate_group":group["candidate_group"],
                                                      "source":nc[0],"target":ce[0]})
    patients = sorted(by_patient,key=lambda p:hashlib.sha256(("venous-general-proposal-v1:"+p).encode()).hexdigest())
    ndev = max(1,round(len(patients)*.2)) if len(patients)>1 else 0
    development = set(patients[:ndev])
    rows, scan_splits = [], {}
    for patient in patients:
        split = "development" if patient in development else "train"
        for pair in sorted(by_patient[patient],key=lambda row:(row["candidate_group"],row["source"],row["target"])):
            if pair["source"] == pair["target"]:
                raise ValueError("Identical source and target ID")
            for scan in (pair["source"],pair["target"]):
                if scan in scan_splits and scan_splits[scan] != split:
                    raise ValueError("Case overlaps provisional splits")
                scan_splits[scan] = split
            rows.append(dict(pair, candidate_patient=patient,provisional_split=split,
                             training_eligible=False,protected_overlap_excluded=False))
    return {"scope":"candidate-only general venous architecture split proposal",
            "source_revision":report["revision"],"source_metadata_hash":report["metadata_sha256"],
            "candidate_patients":len(patients),"candidate_pairs":len(rows),
            "split_candidate_patients":{"train":len(patients)-ndev,"development":ndev},
            "clinical_training_eligible":False,"patient_identity_certified":False,
            "registration_certified":False,"protected_overlap_excluded":False,
            "split_locked_for_experiment":False,"rows":rows}


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--report",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    output=Path(args.output)
    with output.open("x") as stream:
        json.dump(propose(json.loads(Path(args.report).read_text())),stream,indent=2)
