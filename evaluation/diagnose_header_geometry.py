"""Header-only geometry diagnosis. No image voxels, metrics or edits."""
import itertools
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def describe(path):
    img = nib.load(str(path))
    return {'shape': list(img.shape), 'affine': img.affine.tolist(),
            'qform': img.get_qform().tolist(), 'sform': img.get_sform().tolist(),
            'qform_code': int(img.header['qform_code']),
            'sform_code': int(img.header['sform_code']),
            'axis_codes': list(nib.aff2axcodes(img.affine))}


def compare(gt_path, pred_path):
    gt, pred = nib.load(str(gt_path)), nib.load(str(pred_path))
    corners = np.array([(*p, 1) for p in itertools.product(*[(0, n-1) for n in gt.shape])])
    delta = corners @ (pred.affine - gt.affine).T
    return {'same_shape': gt.shape == pred.shape,
            'max_affine_element_delta': float(np.abs(pred.affine - gt.affine).max()),
            'max_corner_displacement_mm': float(np.linalg.norm(delta[:, :3], axis=1).max()),
            'pred_sform_vs_gt_qform_max': float(np.abs(pred.get_sform()-gt.get_qform()).max()),
            'gt_sform_vs_qform_max': float(np.abs(gt.get_sform()-gt.get_qform()).max())}


if __name__ == '__main__':
    root = Path(sys.argv[1])
    cid = 'PanTS_00009232'
    for tag in ('test_ground_truth', 'unetpp_ds', 'unetpp_nods'):
        print(json.dumps({'tag': tag, 'case': cid,
                          'header': describe(root / tag / f'{cid}.nii.gz')}, sort_keys=True))
    for tag in ('unetpp_ds', 'unetpp_nods'):
        failures = []
        paths = sorted((root / tag).glob('PanTS_*.nii.gz'))
        for path in paths:
            result = compare(root / 'test_ground_truth' / path.name, path)
            if not result['same_shape'] or result['max_affine_element_delta'] > 1e-4:
                failures.append({'case': path.name[:-7], **result})
        print(json.dumps({'tag': tag, 'headers_checked': len(paths),
                          'strict_affine_failure_count': len(failures),
                          'first_ten_strict_affine_failures': failures[:10],
                          'largest_corner_displacement_mm': max(
                              (r['max_corner_displacement_mm'] for r in failures), default=0)},
                         sort_keys=True), flush=True)
