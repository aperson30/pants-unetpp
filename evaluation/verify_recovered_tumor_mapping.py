"""Read-only source/index lineage check. No metrics, repairs or new GT.

Run on CPU allocation, with original frozen conversion package on PYTHONPATH.
Full organ-array reconstruction checks converter lineage; only tumor/pancreas
geometry is independently validated here, not other mismatched organ masks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from data_conversion.convert_pants_to_nnunet import CLASS_MAP, merge_labels


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def compare_case(folder, ct_path, gt_path):
    ct = nib.load(str(ct_path))
    gt = nib.load(str(gt_path))
    for organ in CLASS_MAP:
        path = folder / f'{organ}.nii.gz'
        if not path.is_file():
            raise RuntimeError(f'missing original organ: {organ}')
        image = nib.load(str(path))
        if image.shape != ct.shape:
            raise RuntimeError(f'original shape differs: {organ}')
    for organ in ('pancreas', 'pancreatic_lesion'):
        image = nib.load(str(folder / f'{organ}.nii.gz'))
        if not np.allclose(image.affine, ct.affine, rtol=0, atol=1e-4):
            raise RuntimeError(f'target geometry differs: {organ}')
    merged, reference = merge_labels(folder, CLASS_MAP)
    saved = np.asanyarray(gt.dataobj)
    if not np.array_equal(merged, saved):
        raise RuntimeError('saved GT voxels differ from original frozen index merge')
    if not np.allclose(reference.affine, gt.affine, rtol=0, atol=1e-4):
        raise RuntimeError('saved GT affine not explained by original merge reference')
    tumor = np.asanyarray(nib.load(str(folder / 'pancreatic_lesion.nii.gz')).dataobj) > 0
    if not np.array_equal(saved == 28, tumor):
        raise RuntimeError('saved GT tumor voxels differ from original CT-aligned tumor mask')
    return {'source_tumor_equals_saved_gt_tumor': True,
            'source_pancreas_and_tumor_geometry_match_ct': True,
            'all_saved_gt_voxels_equal_original_index_merge': True,
            'saved_gt_affine_matches_first_original_organ': True,
            'saved_gt_affine_matches_ct': bool(np.allclose(gt.affine, ct.affine, rtol=0, atol=1e-4)),
            'ct_sha256': digest(ct_path), 'saved_gt_sha256': digest(gt_path),
            'source_tumor_sha256': digest(folder / 'pancreatic_lesion.nii.gz')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads((args.source_dir / 'source_geometry_inventory.json').read_text())
    input_provenance = json.loads((args.evaluation_dir / 'unetpp_ds/prediction_provenance.json').read_text())
    rows = {}
    for cid in sorted(inventory['cases']):
        ct_paths = list((args.source_dir / 'images').glob(f'**/{cid}/ct.nii.gz'))
        tumor_paths = list((args.source_dir / 'labels').glob(f'**/{cid}/**/pancreatic_lesion.nii.gz'))
        if len(ct_paths) != 1 or len(tumor_paths) != 1:
            raise RuntimeError('ambiguous source paths')
        ct_path = ct_paths[0]; folder = tumor_paths[0].parent
        if digest(ct_path) != input_provenance['inputs'][cid]:
            raise RuntimeError('CT differs from original prediction input')
        for member in inventory['archive_inventory']['PanTSMini_Label.tar.gz']:
            path = args.source_dir / 'labels' / member['member']
            if cid in path.parts and digest(path) != member['sha256']:
                raise RuntimeError('recovered mask changed since archive verification')
        rows[cid] = compare_case(folder, ct_path, args.evaluation_dir / 'test_ground_truth' / f'{cid}.nii.gz')
        print(json.dumps({'case': cid, **rows[cid]}), flush=True)
    print('ALL_RECOVERED_TUMOR_ARRAY_AND_SOURCE_GEOMETRY_CHECKS_PASSED', flush=True)
    print('Other organs with conflicting headers are not certified; no GT repairs performed.', flush=True)
