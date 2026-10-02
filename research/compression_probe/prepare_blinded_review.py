"""CPU-only blinded paired slice packet from EXISTING whole-volume reconstructions.

Private answer key stays remote/outside the public packet. Known-location visual
fidelity review, NOT an unprompted detection study or evidence of clinical harm.
"""
import argparse
import json
from pathlib import Path
import secrets

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def window_slices(labels):
    return np.flatnonzero((labels == 2).sum(axis=(0, 1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    out = root / 'blinded_review_v1'
    private = root / 'blinded_review_v1_private_key.json'
    if private.exists():
        raise RuntimeError('Answer key already exists; never rerandomize a distributed packet')
    out.mkdir(exist_ok=False)
    records = [('120', 'small_followup_120', 'small_results_3290494'),
               ('165', 'small_followup_rank1_attempt2', 'second_results_3290519')]
    rng = secrets.SystemRandom()
    rng.shuffle(records)
    key, public = [], []
    for index, (case, input_folder, result_folder) in enumerate(records, start=1):
        row = json.loads((root / input_folder / 'inputs_manifest.json').read_text())['cases'][0]
        images = [nib.as_closest_canonical(nib.load(p)) for p in
                  (root / result_folder / ('pancreas_'+case) / 'control.nii.gz',
                   root / result_folder / ('pancreas_'+case) / 'reconstruction.nii.gz', row['label'])]
        ref = images[0]
        if any(x.shape != ref.shape or not np.allclose(x.affine, ref.affine, atol=1e-5, rtol=0)
               or x.header.get_xyzt_units()[0] != 'mm' for x in images):
            raise RuntimeError('Review image/mask grid mismatch')
        arrays = [x.get_fdata(dtype=np.float32) if i < 2 else np.asarray(x.dataobj)
                  for i, x in enumerate(images)]
        control, reconstruction, labels = arrays
        slices = window_slices(labels)
        if not 0 < len(slices) <= 16:
            raise RuntimeError('Unexpected slice count; do not silently omit lesion slices')
        spacing = np.linalg.norm(ref.affine[:3, :3], axis=0)
        points = np.argwhere(labels == 2)
        center = np.round((points.min(axis=0)+points.max(axis=0))/2).astype(int)
        half = np.ceil(30 / spacing[:2]).astype(int)
        crop = tuple(slice(max(0, c-h), min(n, c+h+1))
                     for c, h, n in zip(center[:2], half, labels.shape[:2]))
        swapped = rng.choice((False, True))
        pair = (reconstruction, control) if swapped else (control, reconstruction)
        code = f'R{index:03d}'
        filenames = []
        for page, z in enumerate(slices, start=1):
            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            for ax, image, title, zoom in zip(axes,
                    (pair[0], pair[1], pair[0], pair[1]), ('A context', 'B context', 'A detail', 'B detail'),
                    (False, False, True, True)):
                plane = image[:, :, z] if not zoom else image[crop[0], crop[1], z]
                ax.imshow(plane.T, cmap='gray', vmin=-100, vmax=200, origin='lower',
                          aspect=float(spacing[1]/spacing[0]), interpolation='nearest')
                ax.set_title(title); ax.axis('off')
            fig.suptitle(f'{code} — tumor-containing slice {page}/{len(slices)}; same fixed HU window')
            fig.tight_layout()
            filename = f'{code}_slice_{page:02d}.png'
            fig.savefig(out / filename, dpi=160); plt.close(fig)
            filenames.append(filename)
        public.append(dict(case_code=code, pages=filenames,
                           preference=None, confidence=None, lesion_margin_notes=None,
                           visibility_notes=None, uncertain=True))
        key.append(dict(case_code=code, source_case=case, A='reconstruction' if swapped else 'control',
                        B='control' if swapped else 'reconstruction', axial_indices=list(map(int, slices))))
        del arrays, images, control, reconstruction, labels, pair, points
    # Never include this private key in the downloadable reviewer packet/repository.
    with private.open('x') as stream:
        json.dump(key, stream, indent=2)
    with (out / 'review_form.json').open('x') as stream:
        json.dump(dict(protocol='known-location paired visual fidelity; not detection',
                       conditions_hidden=True, reviews=public), stream, indent=2)
    print(json.dumps(dict(packet=str(out), cases=len(public),
                         pages=sum(len(r['pages']) for r in public), private_key_retained_remote=True)), flush=True)


if __name__ == '__main__':
    main()
