"""Independent CPU-only artifact gate; does not change inference or metric math.

Run before scoring/transfer. Compare a saved --reference-manifest after transfer.
No success manifest is emitted until all cases in all cells pass.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from evaluation.predict_and_shrink import file_hash, read_scores
from evaluation.safe_artifacts import retire_report

TAGS = {'unetpp_ds': 'unetpp_ds_on', 'unetpp_nods': 'unetpp_ds_off',
        'default_ds': 'plain_ds_on', 'default_nods': 'plain_ds_off'}


def read_volume(path: Path, *, compact: bool):
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not all(data.shape) or not np.isfinite(data).all():
        raise RuntimeError(f'invalid decoded volume: {path}')
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise RuntimeError(f'invalid affine: {path}')
    if compact:
        valid = ((data == 0) | (data == 28)).all()
    else:
        valid = ((data >= 0) & (data <= 28) & (data == np.floor(data))).all()
    if not valid:
        raise RuntimeError(f'invalid labels: {path}')
    return data, image.affine


def audit(root: Path, expected_cases: int = 901) -> dict:
    expected = {f'PanTS_{i:08d}' for i in range(9001, 9001 + expected_cases)}
    checkpoints = json.loads((root / 'checkpoint_validation_provenance.json').read_text())
    manifest = json.loads((root / 'evaluation_manifest.json').read_text())
    fingerprints = {}; inputs = None
    for tag, key in TAGS.items():
        out = root / tag
        provenance = json.loads((out / 'prediction_provenance.json').read_text())
        if provenance['checkpoint_sha256'] != checkpoints[key]['checkpoint_sha256']:
            raise RuntimeError(f'{tag}: wrong checkpoint')
        if provenance['tumor_class'] != 28 or set(provenance['inputs']) != expected:
            raise RuntimeError(f'{tag}: wrong class/input cases')
        if inputs is None:
            inputs = provenance['inputs']
        if provenance['inputs'] != inputs:
            raise RuntimeError(f'{tag}: different CT inputs')
        files = {p.name[:-7] for p in out.glob('*.nii.gz')}
        scores = read_scores(out / 'max_tumor_probs.csv')
        if files != expected or set(scores) != expected:
            raise RuntimeError(f'{tag}: incomplete or extra mask/score cases')
        fingerprints[f'{tag}/max_tumor_probs.csv'] = file_hash(out / 'max_tumor_probs.csv')
        fingerprints[f'{tag}/prediction_provenance.json'] = file_hash(out / 'prediction_provenance.json')
    gt_dir = root / 'test_ground_truth'
    if {p.name[:-7] for p in gt_dir.glob('*.nii.gz')} != expected:
        raise RuntimeError('wrong GT cases')
    n_positive = 0
    for index, cid in enumerate(sorted(expected), 1):
        gt_path = gt_dir / f'{cid}.nii.gz'
        gt, affine = read_volume(gt_path, compact=False)
        n_positive += int((gt == 28).any())
        fingerprints[f'test_ground_truth/{cid}.nii.gz'] = file_hash(gt_path)
        for tag in TAGS:
            path = root / tag / f'{cid}.nii.gz'
            pred, pred_affine = read_volume(path, compact=True)
            if pred.shape != gt.shape or not np.allclose(pred_affine, affine, rtol=0, atol=1e-4):
                raise RuntimeError(f'{tag}/{cid}: prediction/GT geometry differs')
            fingerprints[f'{tag}/{cid}.nii.gz'] = file_hash(path)
            del pred
        del gt
        if index % 100 == 0:
            print(f'audited {index}/{expected_cases}', flush=True)
    if not 0 < n_positive < expected_cases:
        raise RuntimeError('test GT must contain both positive and negative cases')
    return {'evaluation': manifest, 'cases_per_cell': expected_cases,
            'gt_positive_cases': n_positive, 'gt_negative_cases': expected_cases - n_positive,
            'artifact_sha256': fingerprints}


def verify_grid_report(root: Path, expected_cases: int = 901) -> None:
    report = json.loads((root / 'grid_metrics.json').read_text())
    if report.get('artifact_audit') != audit(root, expected_cases):
        raise RuntimeError('grid report is stale or lacks a current artifact snapshot')
    if set(report.get('metrics', {})) != set(TAGS):
        raise RuntimeError('grid report lacks four metric cells')
    if report.get('project_protocol') is not True:
        raise RuntimeError('grid report lacks the declared project protocol')
    for tag, metrics in report['metrics'].items():
        if metrics.get('n_cases_evaluated') != expected_cases:
            raise RuntimeError(f'{tag}: wrong scored case count')
        protocol = metrics.get('protocol', {})
        if protocol.get('tumor_class') != 28 or protocol.get('T_Sen_connectivity') != '6-neighbor':
            raise RuntimeError(f'{tag}: wrong metric protocol')
        for name in ('DSC_tumor_mean', 'P_Sen', 'T_Sen', 'Spe', 'AUC'):
            value = metrics.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or not 0 <= value <= 1):
                raise RuntimeError(f'{tag}: invalid or missing metric {name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--out-manifest', type=Path)
    parser.add_argument('--reference-manifest', type=Path)
    parser.add_argument('--verify-grid-report', action='store_true')
    args = parser.parse_args()
    if args.verify_grid_report:
        verify_grid_report(args.evaluation_dir)
        print('CURRENT_GRID_REPORT_VERIFIED')
        return
    if not args.out_manifest:
        parser.error('--out-manifest is required unless verifying the grid report')
    # Load reference first: callers may deliberately use the same path for both.
    try:
        reference = json.loads(args.reference_manifest.read_text()) if args.reference_manifest else None
    finally:
        retire_report(args.out_manifest)
    result = audit(args.evaluation_dir)
    if reference is not None and reference != result:
        raise RuntimeError('artifacts differ from the independent saved snapshot')
    temporary = args.out_manifest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.out_manifest)
    print('ALL_FOUR_TEST_ARTIFACTS_AUDITED')


if __name__ == '__main__':
    main()
