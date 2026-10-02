"""CPU parity audit of the pinned official VAE validation transform operations.

Reproduces define_vae_transform(is_train=False, modality='ct', random_aug=False,
spacing_type='original', val_patch_size=None, k=4), NOT diffusion data resizing
or official GPU precision/posterior sampling. No GPU/model inference.
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
    Orientationd, ScaleIntensityRanged, DivisiblePadd, EnsureTyped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    pipeline = Compose([
        LoadImaged(keys='image'), EnsureChannelFirstd(keys='image'),
        Orientationd(keys='image', axcodes='RAS'),
        ScaleIntensityRanged(keys='image', a_min=-1000, a_max=1000,
                             b_min=0, b_max=1, clip=True),
        DivisiblePadd(keys='image', k=4), EnsureTyped(keys='image', dtype=torch.float32)])
    rows = []
    for folder in ('small_followup_120', 'small_followup_rank1_attempt2'):
        manifest = json.loads((args.root / folder / 'inputs_manifest.json').read_text())
        row = manifest['cases'][0]
        image = nib.as_closest_canonical(nib.load(row['image']))
        native = image.get_fdata(dtype=np.float32)
        pads = [(((-n) % 4)//2, ((-n) % 4)-(((-n) % 4)//2)) for n in native.shape]
        expected = np.pad((np.clip(native, -1000, 1000)+1000)/2000, pads)
        actual = pipeline({'image':row['image']})['image'].as_tensor()[0].numpy()
        if expected.shape != actual.shape or not np.isfinite(actual).all():
            raise RuntimeError('Preprocessing shape/finite mismatch')
        result = dict(case=row['case'], native_shape=list(native.shape),
                      spacing_mm=list(map(float, image.header.get_zooms()[:3])),
                      prepared_shape=list(actual.shape),
                      max_abs_normalized_difference=float(np.max(np.abs(actual-expected))),
                      exact_array_equal=bool(np.array_equal(actual, expected)))
        rows.append(result)
        print(json.dumps(result), flush=True)
        del actual, expected, native, image
    report = dict(source_revision='5cb04e82fed71f2fe64a2617ab695be1f5a37fea',
                  source_file='scripts/transforms.py:define_vae_transform',
                  scope='reproduced official validation operations on CPU; NOT full official GPU inference parity',
                  cases=rows)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
