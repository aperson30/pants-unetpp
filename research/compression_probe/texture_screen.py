"""Bounded descriptive texture screen, NOT a detector or REDMOD replication.

Full-volume model outputs already exist. Cropping below is for CPU measurement
only. Fixed source masks and native grids; no resampling or per-arm normalization.
"""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
from scipy.ndimage import gaussian_filter

from prepare_texture_roi import prepare_roi


FEATURES = {
    'firstorder': ['Mean', 'Variance'],
    'glcm': ['Contrast', 'Correlation', 'JointEntropy', 'Idm'],
    'glrlm': ['ShortRunEmphasis', 'LongRunEmphasis'],
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def to_sitk(array, affine):
    """Nibabel xyz/RAS to SimpleITK zyx/LPS, without changing voxel grid."""
    matrix = np.asarray(affine[:3, :3], dtype=float)
    spacing = np.linalg.norm(matrix, axis=0)
    direction = matrix / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError('Unsupported sheared affine')
    flip = np.diag([-1., -1., 1.])
    image = sitk.GetImageFromArray(np.transpose(array, (2, 1, 0)))
    image.SetSpacing(tuple(spacing))
    image.SetDirection(tuple((flip @ direction).ravel()))
    image.SetOrigin(tuple(flip @ affine[:3, 3]))
    return image


def extractor():
    if radiomics.__version__ != 'v3.0.1':
        raise RuntimeError('Expected pinned PyRadiomics v3.0.1')
    obj = featureextractor.RadiomicsFeatureExtractor(
        binWidth=25., normalize=False, distances=[1], force2D=False,
        label=1, voxelArrayShift=0, additionalInfo=False,
        resampledPixelSpacing=None, minimumROIDimensions=3,
    )
    obj.disableAllImageTypes()
    obj.enableImageTypeByName('Original')
    obj.disableAllFeatures()
    obj.enableFeaturesByName(**FEATURES)
    return obj


def extract(obj, array, roi, affine):
    values = obj.execute(to_sitk(array.astype(np.float32), affine),
                         to_sitk(roi.astype(np.uint8), affine))
    result = {k: float(v) for k, v in values.items() if k.startswith('original_')}
    if len(result) != 8 or not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Missing/nonfinite features')
    return result


def self_test():
    obj = extractor()
    array = np.indices((9, 10, 11)).sum(axis=0).astype(np.float32) * 7 - 100
    roi = np.zeros(array.shape, dtype=bool)
    roi[2:-2, 2:-2, 2:-2] = True
    affine = np.diag([1., 2., 3., 1.])
    affine[:3, 3] = (10, 20, 30)
    converted = to_sitk(array, affine)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(converted), array.transpose(2, 1, 0))
    np.testing.assert_allclose(converted.TransformIndexToPhysicalPoint((3, 4, 5)), (-13, -28, 45))
    baseline = extract(obj, array, roi, affine)
    assert baseline == extract(obj, array.copy(), roi.copy(), affine.copy())
    np.testing.assert_allclose(baseline['original_firstorder_Mean'], array[roi].mean(), atol=1e-5)
    np.testing.assert_allclose(baseline['original_firstorder_Variance'], array[roi].var(), rtol=1e-6)
    # A whole-bin HU shift preserves texture under this fixed-width scheme.
    shifted = extract(obj, array + 25, roi, affine)
    for key in baseline:
        expected = baseline[key] + 25 if key.endswith('_Mean') else baseline[key]
        np.testing.assert_allclose(shifted[key], expected, rtol=1e-8, atol=1e-8)
    print('TEXTURE_CONTRACT_TESTS_PASS', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--label', type=Path)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--reconstruction', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    self_test()  # Fail before real extraction if contracts do not hold.
    if args.self_test:
        return
    paths = [args.label, args.control, args.reconstruction]
    if any(p is None for p in paths) or args.output is None:
        parser.error('All input/output paths are required')
    if args.output.exists():
        raise FileExistsError(args.output)
    images = [nib.as_closest_canonical(nib.load(str(p))) for p in paths]
    reference = images[0]
    if any(im.shape != reference.shape or not np.allclose(im.affine, reference.affine,
           atol=1e-5, rtol=0) for im in images):
        raise ValueError('Input grid mismatch')
    if any(im.header.get_xyzt_units()[0] != 'mm' for im in images):
        raise ValueError('Expected mm units')
    spacing = np.linalg.norm(reference.affine[:3, :3], axis=0)
    roi, report = prepare_roi(np.asarray(reference.dataobj), spacing)
    if not roi.any():
        raise ValueError('Empty source ROI')
    # Enough physical halo for the largest Gaussian (truncate=4, sigma=2mm).
    coordinates = np.nonzero(roi)
    halo = np.ceil(8 / spacing).astype(int) + 1
    box = tuple(slice(max(0, int(c.min()) - int(h)),
                      min(n, int(c.max()) + int(h) + 1))
                for c, h, n in zip(coordinates, halo, roi.shape))
    affine = reference.affine.copy()
    affine[:3, 3] = (reference.affine @ np.array([s.start for s in box] + [1]))[:3]
    roi = roi[box]
    arrays = [im.get_fdata(dtype=np.float32)[box].copy() for im in images[1:]]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Nonfinite CT values')
    original, recon = arrays
    bias = float(np.mean(recon[roi].astype(np.float64) - original[roi]))
    arms = {'control': original, 'reconstruction': recon,
            'reconstruction_mean_bias_removed': recon - bias}
    for sigma in (1., 2.):
        arms[f'gaussian_{sigma:g}mm'] = gaussian_filter(original, sigma=sigma/spacing,
                                                       mode='reflect', truncate=4.)
    obj = extractor()
    features = {name: extract(obj, array, roi, affine) for name, array in arms.items()}
    deltas = {name: {k: value - features['control'][k] for k, value in vals.items()}
              for name, vals in features.items() if name != 'control'}
    output = dict(protocol='descriptive pilot; NOT detection, AUC, REDMOD or clinical safety',
                  extractor_version=radiomics.__version__, numpy_version=np.__version__,
                  script_sha256=sha(Path(__file__)),
                  inputs={str(p): sha(p) for p in paths},
                  roi=report, bin_width_hu=25., resampling=False,
                  note='Native anisotropic grid; voxel-neighbor texture is not comparable across cases. '
                       'Gaussian references not noise/PSF matched. Bias removal is secondary, '
                       'uses paired ROI mean; no per-arm rescaling. Settings fixed before extraction.',
                  reconstruction_roi_mean_bias_hu=bias,
                  features=features, absolute_deltas=deltas)
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print('TEXTURE_SCREEN_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
