"""Read-only certificate check before scoring a separate tumor reference.

This does not waive the exact CT replay gate or certify other organ labels.
The production cohort is fixed to 901 cases; smaller cohorts are unit fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.recover_geometry_sources import ARCHIVES, sha

CASES = {f'PanTS_{i:08d}' for i in range(9001, 9902)}


def verify(root: Path, reference: Path, source_audit: Path, *, expected_cases=None):
    expected = CASES if expected_cases is None else set(expected_cases)
    manifest_path = reference / 'tumor_reference_manifest.json'
    manifest_hash = sha(manifest_path)
    certificate = json.loads(manifest_path.read_text())
    audit_hash = sha(source_audit)
    if certificate.get('schema') != 'pants-tumor-reference-v1':
        raise RuntimeError('unsupported reference certificate')
    if certificate.get('source_audit_sha256') != audit_hash:
        raise RuntimeError('source audit identity differs')
    report = json.loads(source_audit.read_text())
    rows = report['cases']
    if report.get('archives') != {name: digest for name, _, digest, _ in ARCHIVES}:
        raise RuntimeError('source archive identities differ')
    for mapping in (rows, certificate['original_gt_sha256'], certificate['reference_sha256']):
        if set(mapping) != expected:
            raise RuntimeError('certificate/audit cohort differs')
    if len(certificate['cases']) != len(expected) or set(certificate['cases']) != expected:
        raise RuntimeError('certificate case list differs')
    snapshots = {str(manifest_path): manifest_hash, str(source_audit): audit_hash}
    evaluation_manifest = root / 'evaluation_manifest.json'
    if sha(evaluation_manifest) != certificate['original_evaluation_manifest_sha256']:
        raise RuntimeError('original evaluation manifest differs')
    snapshots[str(evaluation_manifest)] = sha(evaluation_manifest)
    bound_provenance = certificate['prediction_provenance_sha256']
    if set(bound_provenance) != {'unetpp_ds', 'unetpp_nods'}:
        raise RuntimeError('reference provenance set differs')
    for tag, digest in bound_provenance.items():
        path = root / tag / 'prediction_provenance.json'
        if sha(path) != digest:
            raise RuntimeError('original prediction provenance differs')
        snapshots[str(path)] = digest
        provenance = json.loads(path.read_text())
        if provenance.get('tumor_class') != 28 or set(provenance['inputs']) != expected:
            raise RuntimeError('prediction class/cohort differs')
        for cid, row in rows.items():
            if row.get('tumor_index_equality') is not True or row.get('source_geometry_matches_original_ct') is not True:
                raise RuntimeError('source case lacks complete certification')
            if row['prediction_ct_sha256'] != provenance['inputs'][cid]:
                raise RuntimeError('exact prediction CT identity differs')
    originals = root / 'test_ground_truth'
    labels = reference / 'test_ground_truth'
    for folder in (originals, labels):
        if {p.name[:-7] for p in folder.glob('*.nii.gz')} != expected:
            raise RuntimeError('original/reference file cohort differs')
    for cid in sorted(expected):
        row = rows[cid]
        original = originals / f'{cid}.nii.gz'
        derived = labels / f'{cid}.nii.gz'
        original_hash = sha(original); derived_hash = sha(derived)
        if original_hash != certificate['original_gt_sha256'][cid] or original_hash != row['original_gt_sha256']:
            raise RuntimeError('original GT identity differs')
        if derived_hash != certificate['reference_sha256'][cid]:
            raise RuntimeError('reference identity differs')
        source = nib.load(original); target = nib.load(derived)
        source_data = np.asanyarray(source.dataobj); target_data = np.asanyarray(target.dataobj)
        if list(source.shape) != row['shape'] or source.shape != target.shape or len(source.shape) != 3:
            raise RuntimeError('reference shape differs')
        if not np.isfinite(source_data).all() or not np.isin(source_data, np.arange(29)).all():
            raise RuntimeError('invalid original GT labels')
        if not np.isin(target_data, (0, 28)).all() or not np.array_equal(source_data == 28, target_data == 28):
            raise RuntimeError('reference changed tumor voxels')
        if int((target_data == 28).sum()) != row['tumor_voxels']:
            raise RuntimeError('reference tumor count differs')
        affine = np.asarray(row['prediction_affine'], dtype=float)
        if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise RuntimeError('invalid certified CT geometry')
        if not np.allclose(target.affine, affine, rtol=0, atol=1e-4):
            raise RuntimeError('reference geometry differs')
        snapshots[str(original)] = original_hash; snapshots[str(derived)] = derived_hash
    # Detect changes during this read-only verification as well as stale inputs.
    for path, digest in snapshots.items():
        if sha(Path(path)) != digest:
            raise RuntimeError('reference evidence changed during verification')
    return {'schema': 'pants-tumor-reference-verification-v1', 'cases': sorted(expected),
            'scope': 'class 28 only; no organ certification', 'artifact_sha256': snapshots}
