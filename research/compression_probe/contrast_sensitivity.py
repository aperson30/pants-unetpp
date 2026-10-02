"""Post-hoc CPU diagnostic; NOT a detector or clinical boundary metric.

Tests ring-definition sensitivity and separates lesion/ring intensity shifts.
Both images must be paired, same-grid HU; masks are GT, never predictions.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def contrasts(reference, reconstruction, labels, spacing):
    if reference.shape != reconstruction.shape or reference.shape != labels.shape or reference.ndim != 3:
        raise ValueError('Expected same-grid 3D arrays')
    spacing = np.asarray(spacing, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Invalid spacing')
    if not all(np.isfinite(x).all() for x in (reference, reconstruction, labels)):
        raise ValueError('Nonfinite input')
    if not set(np.unique(labels)) <= {0, 1, 2}:
        raise ValueError('Expected MSD labels')
    components, count = ndi.label(labels == 2, ndi.generate_binary_structure(3, 1))
    if count > 32:
        raise ValueError('Excessive component count for bounded diagnostic')
    rows = []
    for index, box in enumerate(ndi.find_objects(components), start=1):
        # Local distances only: ten-mm measurement halo fully covers <=7mm rings.
        # Model reconstruction has ALREADY run on the whole volume; this does
        # not crop the model input or reduce its context.
        margin = np.ceil(10 / spacing).astype(int)
        roi = tuple(slice(max(0, s.start - m), min(n, s.stop + m))
                    for s, m, n in zip(box, margin, labels.shape))
        lesion = components[roi] == index
        target = labels[roi]
        ref, rec = reference[roi], reconstruction[roi]
        distance = ndi.distance_transform_edt(~lesion, sampling=spacing)
        lesion_delta = float((rec[lesion] - ref[lesion]).mean())
        rings = []
        for lo, hi in ((1, 3), (2, 5), (3, 7)):
            base = (distance >= lo) & (distance <= hi) & (target != 2)
            for tissue, mask in [('all_surrounding', base), ('gt_pancreas_only', base & (target == 1))]:
                n = int(mask.sum())
                row = dict(inner_mm=lo, outer_mm=hi, tissue=tissue, voxels=n)
                if n:
                    a = float(ref[lesion].mean() - ref[mask].mean())
                    b = float(rec[lesion].mean() - rec[mask].mean())
                    row.update(reference_signed_contrast_hu=a,
                               reconstruction_signed_contrast_hu=b,
                               signed_contrast_ratio=b / a if abs(a) > 1e-6 else None,
                               ring_mean_shift_hu=float((rec[mask] - ref[mask]).mean()))
                rings.append(row)
        # Axial slice analysis uses the same 3D 2-5mm surrounding ring. It is
        # descriptive, not a collection of independent statistical samples.
        slices = []
        ring = (distance >= 2) & (distance <= 5) & (target != 2)
        for z in np.flatnonzero(lesion.sum(axis=(0, 1))):
            mask = lesion[:, :, z]
            around = ring[:, :, z]
            row = dict(axial_index=int(z + roi[2].start), tumor_voxels=int(mask.sum()),
                       ring_voxels=int(around.sum()))
            if around.any():
                a = float(ref[:, :, z][mask].mean() - ref[:, :, z][around].mean())
                b = float(rec[:, :, z][mask].mean() - rec[:, :, z][around].mean())
                row.update(reference_signed_contrast_hu=a, reconstruction_signed_contrast_hu=b,
                           signed_contrast_ratio=b / a if abs(a) > 1e-6 else None)
            slices.append(row)
        rows.append(dict(component=index, tumor_voxels=int(lesion.sum()),
                         lesion_mean_shift_hu=lesion_delta, rings=rings, axial_slices=slices))
    return dict(protocol='post-hoc GT-region contrast sensitivity, not recall',
                component_connectivity=6, lesions=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--label', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    images = [nib.as_closest_canonical(nib.load(p)) for p in
              (args.case_dir / 'control.nii.gz', args.case_dir / 'reconstruction.nii.gz', args.label)]
    first = images[0]
    if any(x.shape != first.shape or not np.allclose(x.affine, first.affine, atol=1e-5, rtol=0)
           or x.header.get_xyzt_units()[0] != 'mm' for x in images):
        raise ValueError('Grid/units mismatch')
    matrix = first.affine[:3, :3]
    spacing = np.linalg.norm(matrix, axis=0)
    if not np.allclose((matrix / spacing).T @ (matrix / spacing), np.eye(3), atol=1e-5, rtol=0):
        raise ValueError('Nonorthonormal affine')
    arrays = [x.get_fdata(dtype=np.float32) if i < 2 else np.asarray(x.dataobj)
              for i, x in enumerate(images)]
    result = contrasts(*arrays, spacing)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
