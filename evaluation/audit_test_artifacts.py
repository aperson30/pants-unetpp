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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--out-manifest', type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path)
    args = parser.parse_args()
    result = audit(args.evaluation_dir)
    if args.reference_manifest and json.loads(args.reference_manifest.read_text()) != result:
        raise RuntimeError('artifacts differ from the independent saved snapshot')
    temporary = args.out_manifest.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.out_manifest)
    print('ALL_FOUR_TEST_ARTIFACTS_AUDITED')


if __name__ == '__main__':
    main()
