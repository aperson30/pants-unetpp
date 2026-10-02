"""CPU-only, pinned mask screen; no CT downloads or GPU work.

Select by total GT tumor burden, before reconstruction outcomes. The <=1mL
threshold is an exploratory size criterion, not a clinical-stage definition.
"""
import argparse
import json
from pathlib import Path
import ssl
import urllib.request

import certifi
import nibabel as nib
import numpy as np

from prepare_msd import DATA_REPO, DATA_REV, fetch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=24)
    args = parser.parse_args()
    if not 0 <= args.start <= 128 or not 1 <= args.count <= 48:
        raise RuntimeError('Inventory window exceeds explicit bound')
    # Exclusive output directory prevents clobbering an earlier inventory.
    args.output.mkdir(parents=False, exist_ok=False)
    url = f'https://huggingface.co/api/datasets/{DATA_REPO}/revision/{DATA_REV}?blobs=true'
    with urllib.request.urlopen(url, timeout=60,
            context=ssl.create_default_context(cafile=certifi.where())) as response:
        payload = response.read(4_000_001)
    if len(payload) > 4_000_000:
        raise RuntimeError('Metadata exceeds bound')
    metadata = json.loads(payload)
    if metadata['sha'] != DATA_REV:
        raise RuntimeError('Revision mismatch')
    masks = sorted((r for r in metadata['siblings']
                    if r['rfilename'].startswith('labelsTr/')
                    and r['rfilename'].endswith('.nii.gz')),
                   key=lambda r: r['rfilename'])[args.start:args.start + args.count]
    records = []
    for row in masks:
        name = row['rfilename']
        expected = row['lfs']['sha256']
        if row['lfs']['size'] > 2_000_000:
            raise RuntimeError('Mask download exceeds bound')
        path = args.output / Path(name).name
        fetch(f'https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REV}/{name}',
              path, expected, 2_000_000)
        image = nib.load(path)
        if len(image.shape) != 3 or np.prod(image.shape) > 100_000_000:
            raise RuntimeError('Mask geometry exceeds bound')
        voxel_volume = float(abs(np.linalg.det(image.affine[:3, :3])))
        if not np.isfinite(image.affine).all() or not np.isfinite(voxel_volume) or voxel_volume <= 0:
            raise RuntimeError('Invalid mask affine')
        labels = np.asarray(image.dataobj)
        if image.header.get_xyzt_units()[0] != 'mm' or not set(np.unique(labels)) <= {0, 1, 2}:
            raise RuntimeError('Unexpected units or labels')
        voxels = int(np.count_nonzero(labels == 2))
        volume = float(voxels * voxel_volume)
        record = dict(mask=name, sha256=expected, shape=list(image.shape),
                      tumor_voxels=voxels, tumor_mm3=volume)
        records.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
        del labels, image
    positives = sorted((r for r in records if r['tumor_voxels'] > 0),
                       key=lambda r: (r['tumor_mm3'], r['mask']))
    result = dict(data_repo=DATA_REPO, data_revision=DATA_REV, tumor_label=2,
                  selection='bounded lexicographic mask window; smallest positive total tumor burden',
                  window_start=args.start, window_count=args.count,
                  exploratory_small_cutoff_mm3=1000, inventory=records,
                  candidates=positives[:3],
                  small_candidates=[r for r in positives if r['tumor_mm3'] <= 1000])
    with (args.output / 'inventory.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
