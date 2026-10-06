"""CPU-only exact-byte diagnosis using preserved CT/GT correction metadata.

No edits to original evaluation files. Candidates retain source voxels and are
compared to the immutable prediction-input hash, never to accuracy/Dice.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.recover_geometry_sources import sha

TRANSFORM_FIELDS = ('srow_x', 'srow_y', 'srow_z', 'quatern_b', 'quatern_c',
                    'quatern_d', 'qoffset_x', 'qoffset_y', 'qoffset_z', 'pixdim')


def diagnose(raw, fixed, gt, destination, expected):
    destination.mkdir(parents=True, exist_ok=False)
    raw_image = nib.load(raw); fixed_image = nib.load(fixed); gt_image = nib.load(gt)
    raw_data = np.asanyarray(raw_image.dataobj)
    fixed_data = np.asanyarray(fixed_image.dataobj)
    if not np.array_equal(raw_data, fixed_data):
        raise RuntimeError('frozen replay changed source voxels')
    if gt_image.shape != raw_image.shape:
        raise RuntimeError('preserved GT shape differs from CT')
    # Only investigate tiny differences relative to the independently computed
    # frozen correction, never arbitrary GT orientation or spatial alignment.
    delta = float(np.abs(gt_image.affine - fixed_image.affine).max())
    if delta > 1e-4:
        raise RuntimeError('GT transform not the same corrected CT grid')
    candidates = {}
    header = fixed_image.header.copy()
    for name in TRANSFORM_FIELDS:
        header[name] = gt_image.header[name]
    image = nib.Nifti1Image(fixed_data, None, header)
    path = destination / 'preserved_transform_fields.nii.gz'
    nib.save(image, path)
    candidates[path.name] = path
    path = destination / 'preserved_gt_affine_constructor.nii.gz'
    nib.save(nib.Nifti1Image(raw_data, gt_image.affine, raw_image.header), path)
    candidates[path.name] = path
    rows = {}
    for name, path in candidates.items():
        reloaded = nib.load(path)
        same = np.array_equal(np.asanyarray(reloaded.dataobj), raw_data)
        rows[name] = {'sha256': sha(path), 'matches_original_prediction_sha256': sha(path) == expected,
                      'source_voxels_unchanged': bool(same),
                      'max_affine_delta_from_frozen_replay': float(np.abs(reloaded.affine - fixed_image.affine).max())}
        if not same:
            raise RuntimeError('diagnostic serialization changed source voxels')
    report = {'expected_prediction_sha256': expected,
              'raw_sha256': sha(raw), 'frozen_replay_sha256': sha(fixed),
              'preserved_gt_sha256': sha(gt), 'gt_frozen_affine_max_delta': delta,
              'transform_field_deltas': {name: float(np.max(np.abs(
                  np.asarray(gt_image.header[name], dtype=float) - np.asarray(fixed_image.header[name], dtype=float))))
                  for name in TRANSFORM_FIELDS}, 'candidates': rows}
    with (destination / 'header_replay_report.json').open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--diagnosis-dir', type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    cid = 'PanTS_00009812'
    diagnosis = json.loads((args.diagnosis_dir / 'replay_diagnosis.json').read_text())
    original = json.loads((args.evaluation_dir / 'unetpp_ds/prediction_provenance.json').read_text())
    expected = original['inputs'][cid]
    if diagnosis['cases'][cid]['expected_prediction_sha256'] != expected:
        raise RuntimeError('diagnostic prediction identity changed')
    folder = args.diagnosis_dir / cid
    diagnose(folder / f'{cid}_0000.nii.gz', folder / 'frozen_in_place' / f'{cid}_0000.nii.gz',
             args.evaluation_dir / 'test_ground_truth' / f'{cid}.nii.gz', args.destination, expected)
