"""Read-only four-cell gate for a separately certified tumor reference.

No scores are computed and no files are changed. This does not replace the
source certificate, original-input identity checks, or final metric scoring.
"""
import json
from pathlib import Path

import numpy as np

from evaluation.audit_test_artifacts import TAGS, read_volume
from evaluation.predict_and_shrink import read_scores
from evaluation.recover_geometry_sources import sha
from evaluation.verify_versioned_tumor_reference import verify

PROTOCOL = 'nnUNetv2_predict default fp16, full mirroring, step 0.5, final checkpoint'


def audit(root: Path, reference: Path, source_audit: Path, *, expected_cases=None):
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
    for tag, key in TAGS.items():
        folder = root / tag
        path = folder / 'prediction_provenance.json'
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
        scores = read_scores(folder / 'max_tumor_probs.csv')
        if masks != expected or set(scores) != expected:
            raise RuntimeError(f'{tag}: missing/extra mask or probability cases')
        snapshots[str(path)] = sha(path)
        snapshots[str(folder / 'max_tumor_probs.csv')] = sha(folder / 'max_tumor_probs.csv')
    for cid in sorted(expected):
        shape = tuple(rows[cid]['shape'])
        affine = np.asarray(rows[cid]['prediction_affine'])
        for tag in TAGS:
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
        if {p.name[:-7] for p in (root / tag).glob('*.nii.gz')} != expected:
            raise RuntimeError('prediction cohort changed during audit')
    return {'schema': 'pants-versioned-prediction-audit-v1', 'cases_per_cell': len(expected),
            'gt_positive_cases': positives, 'gt_negative_cases': len(expected) - positives,
            'reference_certificate': certified, 'artifact_sha256': snapshots}
