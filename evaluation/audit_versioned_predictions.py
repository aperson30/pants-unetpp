"""Read-only four-cell gate for a separately certified tumor reference.

No scores are computed and no files are changed. This does not replace the
source certificate, original-input identity checks, or final metric scoring.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.audit_test_artifacts import TAGS, read_volume
from evaluation.predict_and_shrink import read_scores
from evaluation.recover_geometry_sources import sha
from evaluation.verify_versioned_tumor_reference import verify

PROTOCOL = 'nnUNetv2_predict default fp16, full mirroring, step 0.5, final checkpoint'


def audit(root: Path, reference: Path, source_audit: Path, *, expected_cases=None, partial=False):
    certified = verify(root, reference, source_audit, expected_cases=expected_cases)
    expected = set(certified['cases'])
    rows = json.loads(source_audit.read_text())['cases']
    inputs = {cid: row['prediction_ct_sha256'] for cid, row in rows.items()}
    snapshots = dict(certified['artifact_sha256'])
    checkpoint_path = root / 'checkpoint_validation_provenance.json'
    checkpoints = json.loads(checkpoint_path.read_text())
    snapshots[str(checkpoint_path)] = sha(checkpoint_path)
    manifest = json.loads((root / 'evaluation_manifest.json').read_text())
    predictor_hash = manifest['source_hashes']['evaluation/predict_and_shrink.py']
    common_plans = common_dataset = None
    completed = {}
    for tag, key in TAGS.items():
        folder = root / tag
        path = folder / 'prediction_provenance.json'
        if partial and not path.exists():
            if list(folder.glob('*.nii.gz')) or (folder / 'max_tumor_probs.csv').exists():
                raise RuntimeError(f'{tag}: unprovenanced partial predictions')
            completed[tag] = set()
            continue
        provenance = json.loads(path.read_text())
        if provenance['checkpoint_sha256'] != checkpoints[key]['checkpoint_sha256']:
            raise RuntimeError(f'{tag}: wrong checkpoint')
        if provenance.get('tumor_class') != 28 or provenance['inputs'] != inputs:
            raise RuntimeError(f'{tag}: wrong class/CT inputs')
        if provenance.get('predictor_sha256') != predictor_hash or provenance.get('protocol') != PROTOCOL:
            raise RuntimeError(f'{tag}: predictor/inference protocol differs')
        plans = provenance['plans_sha256']; dataset = provenance['dataset_sha256']
        for value in (plans, dataset):
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise RuntimeError(f'{tag}: invalid plans/dataset identity')
        if common_plans is None:
            common_plans, common_dataset = plans, dataset
        if (plans, dataset) != (common_plans, common_dataset):
            raise RuntimeError(f'{tag}: plans/dataset identities differ')
        masks = {p.name[:-7] for p in folder.glob('*.nii.gz')}
        score_path = folder / 'max_tumor_probs.csv'
        scores = read_scores(score_path)
        if partial:
            valid_pairs = masks == set(scores) and masks <= expected
        else:
            valid_pairs = masks == expected and set(scores) == expected
        if not valid_pairs:
            raise RuntimeError(f'{tag}: missing/extra mask or probability cases')
        completed[tag] = masks
        snapshots[str(path)] = sha(path)
        if score_path.exists():
            snapshots[str(score_path)] = sha(score_path)
    for cid in sorted(expected):
        shape = tuple(rows[cid]['shape'])
        affine = np.asarray(rows[cid]['prediction_affine'])
        for tag in TAGS:
            if cid not in completed[tag]:
                continue
            path = root / tag / f'{cid}.nii.gz'
            digest = sha(path)
            data, prediction_affine = read_volume(path, compact=True)
            if data.shape != shape or not np.allclose(prediction_affine, affine, rtol=0, atol=1e-4):
                raise RuntimeError(f'{tag}/{cid}: prediction/certified-CT geometry differs')
            snapshots[str(path)] = digest
            del data
    positives = sum(row['tumor_voxels'] > 0 for row in rows.values())
    if not 0 < positives < len(expected):
        raise RuntimeError('certified cohort needs both positive and negative cases')
    for path, digest in snapshots.items():
        if sha(Path(path)) != digest:
            raise RuntimeError('evaluation evidence changed during prediction audit')
    for tag in TAGS:
        folder = root / tag
        if ({p.name[:-7] for p in folder.glob('*.nii.gz')} != completed[tag]
                or set(read_scores(folder / 'max_tumor_probs.csv')) != completed[tag]
                or (partial and not completed[tag] and
                    (folder / 'prediction_provenance.json').exists() and
                    str(folder / 'prediction_provenance.json') not in snapshots)):
            raise RuntimeError('prediction cohort changed during audit')
    if partial:
        return {'schema': 'pants-partial-prediction-resume-audit-v1',
                'final_scoring_ready': False, 'expected_cases_per_cell': len(expected),
                'completed_cases': {tag: sorted(cases) for tag, cases in completed.items()},
                'remaining_cases': {tag: sorted(expected - cases) for tag, cases in completed.items()},
                'reference_certificate': certified, 'artifact_sha256': snapshots}
    return {'schema': 'pants-versioned-prediction-audit-v1', 'cases_per_cell': len(expected),
            'gt_positive_cases': positives, 'gt_negative_cases': len(expected) - positives,
            'reference_certificate': certified, 'artifact_sha256': snapshots}


if __name__ == '__main__':
    import fcntl
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--partial', action='store_true', help='resume gate only; never a final scoring pass')
    args = parser.parse_args()
    target = args.report.resolve()
    if any(target.is_relative_to(p.resolve()) for p in (args.evaluation_dir, args.reference_dir)):
        raise RuntimeError('audit report must be outside protected input trees')
    if target.exists():
        raise RuntimeError('audit report destination must be fresh')
    with (args.evaluation_dir / '.evaluation.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        result = audit(args.evaluation_dir, args.reference_dir, args.source_audit, partial=args.partial)
        with target.open('x') as output:
            json.dump(result, output, indent=2, allow_nan=False)
    if args.partial:
        print('PARTIAL_PREDICTION_RESUME_AUDIT_PASSED_NOT_FINAL_SCORING',
              {tag: len(cases) for tag, cases in result['completed_cases'].items()}, flush=True)
    else:
        print('COMPLETE_VERSIONED_PREDICTION_AUDIT_PASSED', flush=True)
