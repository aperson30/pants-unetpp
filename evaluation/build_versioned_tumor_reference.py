"""Separate tumor-only reference derivative, gated by the complete source audit.

Never overwrites original combined GT, predictions or manifests. Target voxels
are unchanged; metadata comes from the exact verified prediction CT grid.
This does not certify or repair the 27 non-target organ labels.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.audit_all_source_tumors import CASES
from evaluation.recover_geometry_sources import sha


def write_reference(original, destination, row):
    if sha(original) != row['original_gt_sha256']:
        raise RuntimeError('original GT changed after source audit')
    image = nib.load(original)
    if list(image.shape) != row['shape']:
        raise RuntimeError('original GT shape changed')
    original_data = np.asanyarray(image.dataobj)
    target = original_data == 28
    if int(target.sum()) != row['tumor_voxels']:
        raise RuntimeError('source audit tumor count differs')
    affine = np.asarray(row['prediction_affine'], dtype=np.float64)
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise RuntimeError('invalid verified CT grid')
    reference = nib.Nifti1Image(target.astype(np.uint8) * 28, affine)
    reference.set_data_dtype(np.uint8)
    nib.save(reference, destination)
    reloaded = nib.load(destination)
    data = np.asanyarray(reloaded.dataobj)
    if not np.array_equal(data == 28, target) or not np.isin(data, (0, 28)).all():
        raise RuntimeError('reference serialization changed tumor voxels')
    if not np.allclose(reloaded.affine, affine, rtol=0, atol=1e-4):
        raise RuntimeError('reference serialization changed CT grid')
    if sha(original) != row['original_gt_sha256']:
        raise RuntimeError('original GT changed during reference construction')
    return sha(destination)


def build(root, audit_path, expected_audit_hash, destination):
    if sha(audit_path) != expected_audit_hash:
        raise RuntimeError('source audit hash differs')
    report = json.loads(audit_path.read_text())
    if report.get('diagnostic_only', False) is not False or report.get('ct_identity_mismatches', []):
        raise RuntimeError('diagnostic/mismatched audit is not a source certificate')
    rows = report['cases']
    if set(rows) != CASES:
        raise RuntimeError('source audit is not the complete frozen cohort')
    provenance_paths = [root / tag / 'prediction_provenance.json' for tag in ('unetpp_ds', 'unetpp_nods')]
    provenances = [json.loads(path.read_text()) for path in provenance_paths]
    for provenance in provenances:
        if set(provenance['inputs']) != CASES:
            raise RuntimeError('prediction cohort differs')
        for cid, row in rows.items():
            if row.get('tumor_index_equality') is not True or row.get('source_geometry_matches_original_ct') is not True:
                raise RuntimeError('source audit case did not pass')
            if row['prediction_ct_sha256'] != provenance['inputs'][cid]:
                raise RuntimeError('source audit CT identity differs')
    originals = root / 'test_ground_truth'
    if {path.name[:-7] for path in originals.glob('*.nii.gz')} != CASES:
        raise RuntimeError('original GT cohort differs')
    destination.mkdir(parents=True, exist_ok=False)
    labels = destination / 'test_ground_truth'; labels.mkdir()
    hashes = {}
    for index, cid in enumerate(sorted(CASES), 1):
        hashes[cid] = write_reference(originals / f'{cid}.nii.gz', labels / f'{cid}.nii.gz', rows[cid])
        if index % 100 == 0:
            print(f'REFERENCE_VERIFIED {index}/{len(CASES)}', flush=True)
    if sha(audit_path) != expected_audit_hash:
        raise RuntimeError('source audit changed during construction')
    manifest = {'schema': 'pants-tumor-reference-v1', 'scope': 'class 28 only; other organs not certified',
                'operation': 'unchanged class-28 index mask; exact verified prediction-CT grid metadata; no resampling',
                'source_audit_sha256': expected_audit_hash,
                'original_evaluation_manifest_sha256': sha(root / 'evaluation_manifest.json'),
                'prediction_provenance_sha256': {path.parent.name: sha(path) for path in provenance_paths},
                'original_gt_sha256': {cid: row['original_gt_sha256'] for cid, row in rows.items()},
                'reference_sha256': hashes, 'cases': sorted(CASES),
                'builder_sha256': sha(Path(__file__))}
    # Published only after all cases serialize/reload successfully. Incomplete
    # directories never contain this readiness manifest and must not be scored.
    with (destination / 'tumor_reference_manifest.json').open('x') as output:
        json.dump(manifest, output, indent=2, allow_nan=False)
    print('ALL_VERSIONED_TUMOR_REFERENCES_VERIFIED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--source-audit', type=Path, required=True)
    parser.add_argument('--source-audit-sha256', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    build(args.evaluation_dir, args.source_audit, args.source_audit_sha256, args.destination)
