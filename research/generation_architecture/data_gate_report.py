"""Join saved metadata/mask evidence; do not upgrade hints to training eligibility."""
import argparse
import hashlib
import json
from pathlib import Path


def report(strict, expanded, previous, extension):
    if not previous["archive_scan_complete"] or not previous["requested_targets_complete"]:
        raise ValueError("Previous annotation audit incomplete")
    hashes = {m["metadata_sha256"] for m in (strict, expanded, previous, extension)}
    revisions = {strict["revision"], expanded["revision"], extension["revision"]}
    if len(hashes) != 1 or len(revisions) != 1:
        raise ValueError("Mixed metadata revisions")
    old = previous["inspected_candidate_pancreatic_masks"]
    if set(old).intersection(extension["masks"]):
        raise ValueError("Extension must not overwrite prior evidence")
    masks = old | extension["masks"]
    rows = []
    for group in expanded["candidate_groups"]:
        phases = {}
        for phase in ("noncontrast", "contrast"):
            phases[phase] = [{"case_id": entry["case_id"], "phase_hint": entry["phase_hint"],
                              "foreground_voxels": masks.get(entry["case_id"], {}).get("foreground_voxels")}
                             for entry in group[phase]]
        both = all(any(row["foreground_voxels"] is not None and row["foreground_voxels"] > 0
                       for row in phases[phase]) for phase in phases)
        rows.append({"candidate_group": group["candidate_group"],
                     "candidate_patient": group["candidate_patient"], **phases,
                     "nonempty_pancreatic_masks_both_phases": both,
                     "tumor_correspondence_verified": False, "training_eligible": False})
    patients = {row["candidate_patient"] for row in rows}
    positive_patients = {row["candidate_patient"] for row in rows if row["nonempty_pancreatic_masks_both_phases"]}
    cases = {entry["case_id"] for group in rows for phase in ("noncontrast", "contrast") for entry in group[phase]}
    return {"scope": "candidate label availability; no clinical/registration/split certification",
            "metadata_sha256": next(iter(hashes)), "revision": next(iter(revisions)),
            "candidate_groups": len(rows), "candidate_patients": len(patients), "candidate_series": len(cases),
            "mask_inspections_available": len(cases.intersection(masks)),
            "unknown_mask_cases": sorted(cases - set(masks)),
            "candidate_patients_with_nonempty_both_phase_masks": len(positive_patients),
            "verified_tumor_training_pairs": 0, "frozen_training_manifest": False,
            "gpu_training_launched": False, "groups": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("strict", "expanded", "previous", "extension", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    inputs, provenance = {}, {}
    for name in ("strict", "expanded", "previous", "extension"):
        path = Path(getattr(args, name))
        raw = path.read_bytes()
        inputs[name] = json.loads(raw)
        provenance[name] = {"filename": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
    result = report(**inputs)
    result["input_artifact_provenance"] = provenance
    with open(args.out, "x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}, indent=2))
