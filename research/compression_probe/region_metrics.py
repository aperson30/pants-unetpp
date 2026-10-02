"""CPU-only paired reconstruction diagnostics; these are not clinical realism scores.

Inputs must already be on the SAME grid, in HU, without an implicit resampling.
Compare the model reconstruction with the preprocessing-only control, not merely
with native CT. This keeps clipping/resampling damage separate from VAE damage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


def measure(reference, reconstruction, labels, spacing, tumor_label=28):
    reference = np.asarray(reference, dtype=np.float32)
    reconstruction = np.asarray(reconstruction, dtype=np.float32)
    labels = np.asarray(labels)
    spacing = np.asarray(spacing, dtype=float)
    if reference.ndim != 3 or reference.shape != reconstruction.shape or reference.shape != labels.shape:
        raise ValueError('Expected three same-grid 3D arrays')
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Invalid physical voxel spacing')
    if not all(np.isfinite(a).all() for a in (reference, reconstruction, labels)):
        raise ValueError('Non-finite input')
    if np.any(labels != np.rint(labels)) or labels.min() < 0 or labels.max() > 28:
        raise ValueError('Expected integer PanTS labels 0..28')
    if not isinstance(tumor_label, int) or not 1 <= tumor_label <= 28:
        raise ValueError('Invalid tumor label')
    tumor = labels == tumor_label
    components, count = ndi.label(tumor, ndi.generate_binary_structure(3, 1))
    error = np.abs(reconstruction - reference)
    rows = []
    for index in range(1, count + 1):
        lesion = components == index
        # Physical-distance ring, excluding every tumor component. It is local
        # surrounding tissue, NOT assumed healthy pancreas or radiologist truth.
        outside = ndi.distance_transform_edt(~lesion, sampling=spacing)
        inside = ndi.distance_transform_edt(lesion, sampling=spacing)
        ring = (outside >= 2) & (outside <= 5) & ~tumor
        boundary = (lesion & (inside <= 1)) | (~lesion & (outside <= 1))
        item = {'component': index, 'voxels': int(lesion.sum()),
                'volume_mm3': float(lesion.sum() * np.prod(spacing)),
                'mae_hu': float(error[lesion].mean()),
                'boundary_mae_hu': float(error[boundary].mean()) if boundary.any() else None,
                'ring_voxels': int(ring.sum()),
                'touches_volume_edge': bool(any(np.any(np.take(lesion, pos, axis=axis))
                    for axis in range(3) for pos in (0, -1)))}
        for name, image in [('reference', reference), ('reconstruction', reconstruction)]:
            contrast = float(image[lesion].mean() - image[ring].mean()) if ring.any() else None
            sigma = float(image[ring].std()) if ring.any() else None
            item[name + '_signed_contrast_hu'] = contrast
            item[name + '_cnr'] = contrast / sigma if sigma is not None and sigma > 1e-6 else None
        original = item['reference_signed_contrast_hu']
        current = item['reconstruction_signed_contrast_hu']
        item['contrast_ratio'] = current / original if original is not None and abs(original) > 1e-6 else None
        rows.append(item)
    return {'protocol': 'same-grid HU reconstruction diagnostics, not clinical realism',
            'component_connectivity': 6, 'tumor_label': tumor_label,
            'tumor_voxels': int(tumor.sum()), 'lesions': rows,
            'whole_image_mae_hu': float(error.mean())}


def load_same_grid(paths):
    images = [nib.load(str(path)) for path in paths]
    first = images[0]
    if len(first.shape) != 3:
        raise ValueError('Expected 3D images')
    for image in images[1:]:
        if image.shape != first.shape or not np.allclose(image.affine, first.affine, rtol=0, atol=1e-5):
            raise ValueError('Grid mismatch: no implicit alignment allowed')
    matrix = first.affine[:3, :3]
    spacing = np.linalg.norm(matrix, axis=0)
    axes = matrix / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError('Sheared or degenerate affine unsupported by physical-distance metric')
    if any(image.header.get_xyzt_units()[0] != 'mm' for image in images):
        raise ValueError('Physical-distance metrics require explicit mm units')
    return [np.asarray(image.dataobj) for image in images], spacing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True,
                        help='Preprocessing-only control on reconstruction grid, in HU')
    parser.add_argument('--reconstruction', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    arrays, spacing = load_same_grid([args.reference, args.reconstruction, args.labels])
    result = measure(*arrays, spacing)
    result['inputs'] = {key: str(getattr(args, key).resolve())
                        for key in ('reference', 'reconstruction', 'labels')}
    # Exclusive creation prevents overwriting earlier evidence. Failed writes
    # cannot be mistaken for JSON because readers must parse the complete file.
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
