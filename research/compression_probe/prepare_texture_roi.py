"""CPU-only MSD parenchymal ROI preparation; NOT radiomics or cancer detection.

Fixed source-label mask shared by all image conditions. Distances are voxel-
center Euclidean distances in mm, not an anatomical boundary measurement.
No image intensity or reconstruction output is used to choose the ROI.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def prepare_roi(labels, spacing, tumor_margin_mm=5., edge_margin_mm=2.):
    labels = np.asarray(labels)
    spacing = np.asarray(spacing, dtype=float)
    if labels.ndim != 3 or not labels.size or min(labels.shape) == 0:
        raise ValueError('Expected nonempty 3D mask')
    if not np.isfinite(labels).all() or not set(np.unique(labels)) <= {0, 1, 2}:
        raise ValueError('Expected integer MSD labels 0/1/2')
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Invalid physical spacing')
    if not np.isfinite([tumor_margin_mm, edge_margin_mm]).all() or min(tumor_margin_mm, edge_margin_mm) < 0:
        raise ValueError('Invalid margins')
    organ = labels > 0
    # Explicit zero padding prevents undefined all-foreground EDT behavior.
    inner = ndi.distance_transform_edt(np.pad(organ, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    roi = (labels == 1) & (inner > edge_margin_mm)
    if np.any(labels == 2):
        distance = ndi.distance_transform_edt(labels != 2, sampling=spacing)
        roi &= distance > tumor_margin_mm
    count = int(roi.sum())
    report = dict(protocol='fixed source-mask parenchymal ROI feasibility only; NOT healthy-tissue truth',
                  tumor_margin_mm=float(tumor_margin_mm), edge_margin_mm=float(edge_margin_mm),
                  spacing_mm=spacing.tolist(), organ_voxels=int(organ.sum()),
                  source_pancreas_voxels=int((labels == 1).sum()), tumor_voxels=int((labels == 2).sum()),
                  roi_voxels=count, roi_volume_mm3=float(count * np.prod(spacing)),
                  status='nonempty' if count else 'empty-do-not-extract',
                  duct_mask_available=False)
    if count:
        positions = np.nonzero(roi)
        report['roi_extent_voxels'] = [int(p.max()-p.min()+1) for p in positions]
    return roi, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    image = nib.load(str(args.label))
    if image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('Expected mm units')
    matrix = image.affine[:3, :3]
    spacing = np.linalg.norm(matrix, axis=0)
    if not np.isfinite(image.affine).all() or np.any(spacing <= 0):
        raise ValueError('Invalid affine')
    if not np.allclose((matrix/spacing).T @ (matrix/spacing), np.eye(3), atol=1e-5, rtol=0):
        raise ValueError('Sheared affine is unsupported')
    _, report = prepare_roi(np.asarray(image.dataobj), spacing)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
