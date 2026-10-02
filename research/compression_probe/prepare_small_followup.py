"""Prepare one preselected small case; CPU/download only, never submits jobs."""
import argparse
import json
from pathlib import Path
import shutil
import ssl
import urllib.request

import certifi
import nibabel as nib
import numpy as np

from prepare_msd import DATA_REPO, DATA_REV, MODEL_HASH, MODEL_REV, fetch, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = []
    for folder in ('small_inventory_24', 'small_inventory_next48', 'small_inventory_third48'):
        inventory = json.loads((root / folder / 'inventory.json').read_text())
        if inventory['data_revision'] != DATA_REV or inventory['tumor_label'] != 2:
            raise RuntimeError('Inventory provenance changed')
        for row in inventory['inventory']:
            rows.append(dict(row, label=str(root / folder / Path(row['mask']).name)))
    if len(rows) != 120 or len({r['mask'] for r in rows}) != 120:
        raise RuntimeError('Expected 120 unique prespecified masks')
    positive = sorted((r for r in rows if r['tumor_voxels'] > 0),
                      key=lambda r: (r['tumor_mm3'], r['mask']))
    selected = positive[0]
    if selected['tumor_mm3'] > 1000:
        raise RuntimeError('No small candidate under prespecified cutoff')
    if shutil.disk_usage(root).free < 1024**3:
        raise RuntimeError('Need 1GiB headroom')
    output = root / 'small_followup_120'
    output.mkdir(exist_ok=False)
    with urllib.request.urlopen(
            f'https://huggingface.co/api/datasets/{DATA_REPO}/revision/{DATA_REV}?blobs=true',
            timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response:
        payload = response.read(4_000_001)
    if len(payload) > 4_000_000:
        raise RuntimeError('Metadata exceeds bound')
    metadata = json.loads(payload)
    if metadata['sha'] != DATA_REV:
        raise RuntimeError('Metadata revision mismatch')
    image_name = selected['mask'].replace('labelsTr/', 'imagesTr/', 1)
    info = next(r for r in metadata['siblings'] if r['rfilename'] == image_name)['lfs']
    if info['size'] > 80_000_000:
        raise RuntimeError('CT exceeds download bound')
    image_path = output / Path(image_name).name
    fetch(f'https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REV}/{image_name}',
          image_path, info['sha256'], 80_000_000)
    label_path = Path(selected['label'])
    if sha(label_path) != selected['sha256']:
        raise RuntimeError('Mask hash mismatch')
    ct, label = nib.load(image_path), nib.load(label_path)
    if ct.shape != label.shape or not np.allclose(ct.affine, label.affine, atol=1e-5, rtol=0):
        raise RuntimeError('CT/mask geometry mismatch')
    if ct.header.get_xyzt_units()[0] != 'mm' or label.header.get_xyzt_units()[0] != 'mm':
        raise RuntimeError('Expected millimeter geometry')
    canonical = nib.as_closest_canonical(ct)
    matrix = canonical.affine[:3, :3]
    spacing = np.linalg.norm(matrix, axis=0)
    if not np.isfinite(matrix).all() or np.any(spacing <= 0) or not np.allclose(
            (matrix / spacing).T @ (matrix / spacing), np.eye(3), atol=1e-5, rtol=0):
        raise RuntimeError('Invalid/sheared affine')
    model = root / 'inputs' / 'autoencoder_v1.pt'
    if sha(model) != MODEL_HASH:
        raise RuntimeError('Model hash mismatch')
    case = Path(image_name).name.removeprefix('pancreas_').removesuffix('.nii.gz')
    row = dict(case=case, image=str(image_path), image_sha256=info['sha256'],
               label=str(label_path), label_sha256=selected['sha256'],
               shape=list(ct.shape), tumor_voxels=selected['tumor_voxels'],
               tumor_mm3=selected['tumor_mm3'])
    manifest = dict(model=str(model), model_sha256=MODEL_HASH, model_revision=MODEL_REV,
                    data_repo=DATA_REPO, data_revision=DATA_REV, tumor_label=2,
                    selection='smallest positive total GT tumor burden among first 120 lexicographic masks; <=1mL',
                    cases=[row])
    with (output / 'inputs_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    print(json.dumps(manifest, indent=2), flush=True)
    print('PREPARATION_COMPLETE_NO_GPU_JOB', flush=True)


if __name__ == '__main__':
    main()
