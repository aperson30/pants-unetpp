"""Read-only partial evaluation audit; never declares the grid complete.

Run on a CPU allocation, not a login node. Does not score test outcomes, load
checkpoints, repair files, alter source manifests, or change inference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.audit_test_artifacts import TAGS, read_volume
from evaluation.predict_and_shrink import file_hash, read_scores


def audit_partial(root: Path, frozen: Path, expected_cases: int = 901) -> dict:
    expected = {f'PanTS_{i:08d}' for i in range(9001, 9001 + expected_cases)}
    manifest = json.loads((root / 'evaluation_manifest.json').read_text())
    if (frozen / '.frozen_commit').read_text().strip() != manifest['first_evaluation_commit']:
        raise RuntimeError('wrong frozen evaluation revision')
    if not manifest['source_hashes']:
        raise RuntimeError('missing source fingerprints')
    for name, digest in manifest['source_hashes'].items():
        path = (frozen / name).resolve()
        if not path.is_relative_to(frozen.resolve()) or file_hash(path) != digest:
            raise RuntimeError(f'changed evaluation source: {name}')
    checkpoints = json.loads((root / 'checkpoint_validation_provenance.json').read_text())
    gt_dir = root / 'test_ground_truth'
    if {p.name[:-7] for p in gt_dir.glob('*.nii.gz')} != expected:
        raise RuntimeError('incorrect saved GT case membership')
    cells = {}; common_inputs = None; fingerprints = {}
    for tag, key in TAGS.items():
        out = root / tag
        if not out.exists():
            cells[tag] = {'saved_pairs': 0, 'remaining': expected_cases, 'started': False}
            continue
        provenance = json.loads((out / 'prediction_provenance.json').read_text())
        if provenance['checkpoint_sha256'] != checkpoints[key]['checkpoint_sha256']:
            raise RuntimeError(f'{tag}: checkpoint provenance differs')
        if provenance['predictor_sha256'] != manifest['source_hashes']['evaluation/predict_and_shrink.py']:
            raise RuntimeError(f'{tag}: predictor differs')
        if provenance['tumor_class'] != 28 or set(provenance['inputs']) != expected:
            raise RuntimeError(f'{tag}: wrong class/input membership')
        if common_inputs is None:
            common_inputs = provenance['inputs']
        if common_inputs != provenance['inputs']:
            raise RuntimeError(f'{tag}: input fingerprints differ across cells')
        scores = read_scores(out / 'max_tumor_probs.csv')
        files = {p.name[:-7] for p in out.glob('*.nii.gz')}
        if not set(scores) <= expected or not files <= expected:
            raise RuntimeError(f'{tag}: extra cases')
        # A crash between publication of mask and score is safe to recompute,
        # but must be explicitly listed, not counted as a completed pair.
        paired = files & set(scores)
        for cid in sorted(paired):
            gt, affine = read_volume(gt_dir / f'{cid}.nii.gz', compact=False)
            pred, pred_affine = read_volume(out / f'{cid}.nii.gz', compact=True)
            if gt.shape != pred.shape or not np.allclose(affine, pred_affine, rtol=0, atol=1e-4):
                raise RuntimeError(f'{tag}/{cid}: geometry differs')
            fingerprints[f'{tag}/{cid}.nii.gz'] = file_hash(out / f'{cid}.nii.gz')
            del gt, pred
        fingerprints[f'{tag}/max_tumor_probs.csv'] = file_hash(out / 'max_tumor_probs.csv')
        cells[tag] = {'saved_pairs': len(paired), 'remaining': expected_cases - len(paired),
                      'started': True, 'mask_without_score': sorted(files - set(scores)),
                      'score_without_mask': sorted(set(scores) - files)}
        print(f'{tag}: {len(paired)}/{expected_cases} saved pairs checked', flush=True)
    return {'scope': 'partial resume integrity, NOT complete grid or metric results',
            'training_commit': manifest['training_commit'],
            'evaluation_commit': manifest['first_evaluation_commit'],
            'cells': cells, 'saved_prediction_sha256': fingerprints}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--frozen-evaluation', type=Path, required=True)
    args = parser.parse_args()
    result = audit_partial(args.evaluation_dir, args.frozen_evaluation)
    print('PARTIAL_RESUME_AUDIT_JSON=' + json.dumps(result, sort_keys=True))
