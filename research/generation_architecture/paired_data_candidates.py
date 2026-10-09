"""Bounded public metadata inventory, not verified phase pairs or clinical labels.

Persist public CV file IDs and local anonymous group numbers only; never patient
IDs, dates, accession values or report text. No CT/mask download or GPU use.
"""
import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
from collections import defaultdict


REVISION = "c15802c28cb31b71dee2e680319d522b3bc4cbbf"
URL = f"https://huggingface.co/datasets/BodyMaps/CancerVerse/resolve/{REVISION}/CancerVerse_dataset_metadata.csv"
NC = {"non-contrast", "noncontrast", "non contrast", "nc"}
CE = {"arterial", "arterial_early", "arterial_late", "venous", "portal_venous", "portal venous", "delayed"}
MISSING = {"", "na", "n/a", "nan", "none", "unknown"}


def inventory(raw, grouping="patient_accession_date"):
    if grouping not in ("patient_accession_date", "patient_date"):
        raise ValueError("Unknown candidate grouping")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    required = {"CancerVerse ID", "Patient ID", "Encrypted Accession Number", "exam_date", "phase"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Required identity/phase columns absent")
    groups = defaultdict(list)
    case_ids = set()
    for row in reader:
        case = row["CancerVerse ID"].strip()
        if not re.fullmatch(r"CV_[0-9]{8}", case) or case in case_ids:
            raise ValueError("Malformed or duplicate public case ID")
        case_ids.add(case)
        columns = ("Patient ID", "Encrypted Accession Number", "exam_date") if grouping == "patient_accession_date" else ("Patient ID", "exam_date")
        key = tuple(row[k].strip() for k in columns)
        if any(value.casefold() in MISSING for value in key):
            continue
        phase = row["phase"].strip().casefold()
        groups[key].append({"case_id": case, "phase_hint": phase})
    candidates = [(key, values) for key, values in groups.items()
                  if any(r["phase_hint"] in NC for r in values)
                  and any(r["phase_hint"] in CE for r in values)]
    patients = {patient: i + 1 for i, patient in enumerate(sorted({key[0] for key, _ in candidates}))}
    result = []
    for index, (key, values) in enumerate(sorted(candidates), 1):
        result.append({"candidate_group": f"group_{index:03d}",
                       "candidate_patient": f"patient_{patients[key[0]]:03d}",
                       "noncontrast": sorted((r for r in values if r["phase_hint"] in NC), key=lambda r: r["case_id"]),
                       "contrast": sorted((r for r in values if r["phase_hint"] in CE), key=lambda r: r["case_id"]),
                       "patient_linkage_verified": False, "phase_acquisition_verified": False,
                       "registration_verified": False, "tumor_annotations_verified": False,
                       "training_eligible": False})
    return {"revision": REVISION, "metadata_sha256": hashlib.sha256(raw).hexdigest(), "grouping": grouping,
            "candidate_groups": result, "scope": "metadata hints only; all training eligibility false",
            "raw_identifiers_or_reports_saved": False, "CT_or_masks_downloaded": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--grouping", choices=("patient_accession_date", "patient_date"), default="patient_accession_date")
    args = parser.parse_args()
    with urllib.request.urlopen(URL, timeout=30) as response:
        raw = response.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError("Metadata exceeds bounded cap")
    result = inventory(raw, grouping=args.grouping)
    with open(args.out, "x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps({"candidate_groups": len(result["candidate_groups"]),
                      "verified_training_pairs": 0, "raw_identifiers_saved": False}))
