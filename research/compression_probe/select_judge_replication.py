"""Lock two additional size-biased feasibility cases before any predictions.

Uses existing pinned eligibility and clinical snapshot for patient uniqueness.
Clinical workbook and patient IDs never appear in output or public artifacts.
This is not a representative cohort and has no clinical-recall endpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def select(audit):
    eligible = [c for c in audit['candidates'] if c['fold'] == 4
                and c['reference'] == 'histopathology'
                and c['study'] != '100226_00001'
                and not c['archive']['encrypted']
                and c['archive']['compression'] == 0
                and c['archive']['compressed'] <= 60_000_000]
    selected = sorted(eligible, key=lambda c: (c['archive']['compressed'], c['study']))[:2]
    if len(selected) != 2 or len({c['study'] for c in selected}) != 2:
        raise ValueError('Need two distinct eligible cases; do not substitute after outcomes')
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eligibility', type=Path, required=True)
    parser.add_argument('--clinical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.eligibility.read_text())
    if hashlib.sha256(args.clinical.read_bytes()).hexdigest() != audit['summary']['provenance']['clinical_sha256']:
        raise ValueError('Clinical source hash mismatch')
    selected = select(audit)
    clinical = pd.read_excel(args.clinical, sheet_name='Sheet1')
    studies = [c['study'] for c in selected] + ['100226_00001']
    rows = clinical[clinical.PANORAMA_study_id.isin(studies)]
    if len(rows) != 3 or rows.PANORAMA_study_id.nunique() != 3 or rows.PANORAMA_patient_id.isna().any() or rows.PANORAMA_patient_id.nunique() != 3:
        raise ValueError('Pilot and replication must be three distinct patients')
    result = dict(summary=audit['summary'], candidates=selected,
                  selection_rule='Two smallest stored batch1 histopathology PDAC cases in fold4 under60MB, excluding pilot; size-biased feasibility only',
                  patient_unique_with_pilot=True, no_outcomes_used=True,
                  eligibility_sha256=hashlib.sha256(args.eligibility.read_bytes()).hexdigest(),
                  failure_rule='Retain failures/original misses; no outcome-driven replacement',
                  endpoint='Paired continuous tumor score; no clinical threshold/recall')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('SELECTION_LOCKED', ','.join(c['study'] for c in selected), flush=True)


if __name__ == '__main__':
    main()
