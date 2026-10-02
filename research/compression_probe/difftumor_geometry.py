"""Modern equivalents of published image transforms; raw-head diagnostic only.

Published final tumor masks additionally require an unverified organ-pseudo
producer. Do not represent these raw logits/probabilities as that final detector.
No label is accepted by preprocessing or inference.
"""
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (
    Compose, Invertd, Orientationd, ScaleIntensityRanged, Spacingd, SpatialPadd,
)


def image_transforms():
    return Compose([
        Orientationd('image', axcodes='RAS', labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))),
        Spacingd('image', pixdim=(1., 1., 1.), mode='bilinear'),
        ScaleIntensityRanged('image', a_min=-175., a_max=250.,
                             b_min=0., b_max=1., clip=True),
        SpatialPadd('image', spatial_size=(96, 96, 96), mode='minimum'),
    ])


def prepare(array, affine):
    array = np.asarray(array, dtype=np.float32)
    affine = np.asarray(affine, dtype=np.float64)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError('Require finite full 3D image')
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError('Invalid affine')
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if (spacing <= 0).any() or not np.allclose(
            (affine[:3, :3] / spacing).T @ (affine[:3, :3] / spacing),
            np.eye(3), atol=1e-5, rtol=0):
        raise ValueError('Shear/degenerate affine: no silent geometry rescue')
    transform = image_transforms()
    data = transform({'image': MetaTensor(torch.from_numpy(array.copy())[None], affine=affine)})
    return data, transform


def invert_logits(logits, data, transform, native_shape, native_affine):
    if logits.ndim != 4 or logits.shape[0] != 3 or tuple(logits.shape[1:]) != tuple(data['image'].shape[1:]):
        raise ValueError('Logit grid/channel mismatch')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite logits')
    inverse = Invertd('pred', transform=transform, orig_keys='image',
                     nearest_interp=False, to_tensor=True)
    # Modern Invertd chooses its metadata path from the prediction's type.
    # A plain Tensor silently selects legacy image_transforms dictionary keys.
    result = inverse({**data, 'pred': MetaTensor(logits.cpu())})['pred']
    if tuple(result.shape) != (3,) + tuple(native_shape) or not np.allclose(
            result.affine.cpu().numpy(), native_affine, atol=1e-5, rtol=0):
        raise ValueError('Native grid restoration failed')
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite native-grid logits')
    return result
