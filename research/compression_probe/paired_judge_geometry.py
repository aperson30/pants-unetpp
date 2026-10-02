"""Native/canonical geometry and GT-only diagnostics for one-case pilot."""
import nibabel as nib
import numpy as np


def restore_native(canonical_array, canonical_image, native_image):
    if canonical_array.shape != canonical_image.shape or not np.isfinite(canonical_array).all():
        raise ValueError('Invalid canonical reconstruction')
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(canonical_image.affine),
        nib.orientations.io_orientation(native_image.affine))
    output = nib.orientations.apply_orientation(canonical_array, transform)
    restored_affine = canonical_image.affine @ nib.orientations.inv_ornt_aff(transform, canonical_array.shape)
    if output.shape != native_image.shape or not np.allclose(restored_affine, native_image.affine, atol=1e-5, rtol=0):
        raise ValueError('Reconstruction cannot be restored without resampling')
    return output.astype(np.float32, copy=False)


def score_tumor_maps(tumor, raw, masked, candidates, crop_fraction):
    """Descriptive one-case scores, NOT sensitivity or clinical thresholds."""
    if tumor.dtype != np.bool_ or tumor.ndim != 3 or not tumor.any():
        raise ValueError('Require nonempty boolean tumor GT')
    if not 0 <= crop_fraction <= 1:
        raise ValueError('Invalid crop fraction')
    for array in (raw, masked, candidates):
        if array.shape != tumor.shape or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError('Invalid or misaligned probability map')
    return dict(tumor_voxels=int(tumor.sum()), tumor_fraction_in_crop=float(crop_fraction),
                tumor_mean_raw_probability=float(raw[tumor].mean()),
                tumor_max_raw_probability=float(raw[tumor].max()),
                tumor_mean_masked_probability=float(masked[tumor].mean()),
                candidate_confidence_on_tumor=float(candidates[tumor].max()),
                tumor_voxels_with_candidate=int(np.count_nonzero(candidates[tumor])),
                patient_max_candidate_score=float(candidates.max()),
                interpretation='continuous diagnostic scores; no clinical detection threshold fixed here')
