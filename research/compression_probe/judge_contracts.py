"""CPU guards and diagnostics, NOT a replacement PANORAMA inference pipeline.

Arrays here are ZYX. Physical affine validation accepts a separately constructed
XYZ-index-to-physical-mm matrix. Actual SimpleITK resampling/cropping parity must
still be tested in the selected inference runtime.
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation


def require_same_grid(shape_a, affine_a, shape_b, affine_b):
    a, b = np.asarray(affine_a), np.asarray(affine_b)
    if tuple(shape_a) != tuple(shape_b) or a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError('Physical grid shape mismatch')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite geometry')
    for matrix in (a, b):
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
            raise ValueError('Not an affine physical grid')
        if abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
            raise ValueError('Singular geometry')
    if not np.allclose(a, b, atol=1e-5, rtol=0):
        raise ValueError('Physical grid mismatch')


def crop_slices(shape_zyx, coordinates):
    if len(shape_zyx) != 3:
        raise ValueError('Expected 3D ZYX shape')
    slices = []
    for axis, size in zip('zyx', shape_zyx):
        start, finish = coordinates[f'{axis}_start'], coordinates[f'{axis}_finish']
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               for v in (start, finish)):
            raise ValueError('Bounds must be integers, not rounded silently')
        if not 0 <= start < finish <= size:
            raise ValueError('Empty or out-of-range crop')
        slices.append(slice(start, finish))
    return tuple(slices)


def expand_map(prediction_zyx, shape_zyx, coordinates):
    slices = crop_slices(shape_zyx, coordinates)
    prediction = np.asarray(prediction_zyx)
    expected = tuple(s.stop - s.start for s in slices)
    if prediction.shape != expected or not np.isfinite(prediction).all():
        raise ValueError('Prediction shape/nonfinite mismatch')
    if np.any((prediction < 0) | (prediction > 1)):
        raise ValueError('Expected probabilities in [0,1]')
    output = np.zeros(shape_zyx, dtype=np.float32)
    output[slices] = prediction
    return output


def postprocess_probability(probability_zyx, segmentation_zyx):
    """Replicate audited mask operation; does NOT extract lesion candidates."""
    probability, segmentation = np.asarray(probability_zyx), np.asarray(segmentation_zyx)
    if probability.ndim != 3 or probability.shape != segmentation.shape:
        raise ValueError('Expected aligned 3D probability and segmentation')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid probabilities')
    if not np.isfinite(segmentation).all() or not np.equal(segmentation, np.floor(segmentation)).all():
        raise ValueError('Invalid segmentation labels')
    # Exact dataset label set must additionally be checked against released plans.
    mask = binary_dilation(np.isin(segmentation, [1, 4, 5]),
                           structure=np.ones((5, 5, 5), dtype=bool))
    output = probability.astype(np.float32, copy=True)
    output[~mask] = 0
    return output, mask


def tumor_crop_fraction(tumor_mask_zyx, coordinates):
    """Scoring only: must never drive crop selection or inference."""
    tumor = np.asarray(tumor_mask_zyx)
    if tumor.dtype != np.bool_ or tumor.ndim != 3 or not tumor.any():
        raise ValueError('Expected a nonempty 3D boolean tumor mask')
    return float(tumor[crop_slices(tumor.shape, coordinates)].sum() / tumor.sum())


def reserve_arm_directory(path):
    """Atomic leaf reservation: refuses existing output, even an empty folder."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
