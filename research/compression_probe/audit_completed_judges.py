"""CPU-only audit of frozen outputs. No inference or outcome-based selection.

Candidate cutoff diagnostic is posthoc and specific to audited RGA0.3.4
dynamic-fast defaults. It is not a locked clinical detection threshold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def audit_case(output, label_path):
    completion = json.loads((output/'completion.json').read_text())
    if completion.get('completed') is not True:
        raise ValueError('Require complete producer output')
    tumor = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))) == 1
    rows = []
    reports = {r['arm']: r for r in completion['results']}
    if set(reports) != {'native', 'control', 'reconstruction'} or not tumor.any():
        raise ValueError('Require three arms and nonempty tumor')
    for arm in ('native', 'control', 'reconstruction'):
        with np.load(output/arm/'tumor_diagnostic_maps.npz', allow_pickle=False) as maps:
            raw, masked, candidates = (maps[k] for k in ('raw_pdac', 'masked_pdac', 'candidates'))
            if any(x.shape != tumor.shape or not np.isfinite(x).all() for x in (raw, masked, candidates)):
                raise ValueError('Invalid saved diagnostic maps')
            row = dict(arm=arm, tumor_mean_raw_probability=float(raw[tumor].mean()),
                       tumor_max_raw_probability=float(raw[tumor].max()),
                       candidate_confidence_on_tumor=float(candidates[tumor].max()),
                       tumor_voxels_with_candidate=int((candidates[tumor] > 0).sum()),
                       patient_max_candidate_score=float(candidates.max()),
                       masked_global_max=float(masked.max()),
                       dynamic_fast_cutoff=float(masked.max()/2.5),
                       tumor_max_below_dynamic_cutoff=bool(masked[tumor].max() < masked.max()/2.5))
            for key in ('tumor_mean_raw_probability', 'tumor_max_raw_probability',
                        'candidate_confidence_on_tumor', 'patient_max_candidate_score'):
                if not np.isclose(row[key], reports[arm][key], rtol=1e-6, atol=1e-8):
                    raise ValueError('Saved map/producer report mismatch')
            if row['tumor_voxels_with_candidate'] != reports[arm]['tumor_voxels_with_candidate']:
                raise ValueError('Candidate overlap count mismatch')
            rows.append(row)
    original, control, reconstruction = rows
    return dict(completed=True, clinical_recall=False, arms=rows,
                native_control_tumor_scores_identical=all(original[k] == control[k] for k in
                    ('tumor_mean_raw_probability', 'tumor_max_raw_probability', 'candidate_confidence_on_tumor')),
                reconstructed_to_control_mean_ratio=reconstruction['tumor_mean_raw_probability']/control['tumor_mean_raw_probability']
                    if control['tumor_mean_raw_probability'] > 0 else None,
                original_has_gt_overlapping_candidate=original['tumor_voxels_with_candidate'] > 0,
                native_reconstruction_crop_identical=reports['native']['crop_bounds']==reports['reconstruction']['crop_bounds'],
                caveat='Nonzero candidate overlap is not clinical detection; dynamic cutoff explanation is posthoc')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--producer-output', type=Path, required=True)
    parser.add_argument('--label', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_case(args.producer_output, args.label)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('SAVED_MAP_AUDIT_COMPLETE_NOT_RECALL', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
