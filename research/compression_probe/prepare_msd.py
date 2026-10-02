"""Bounded CPU preparation: four masks, at most two CTs, ONE VAE weight.

Uses a pinned third-party MSD mirror; hashes prove mirror identity, not original
MSD archive equivalence. This is exploratory, not an official benchmark claim.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
import urllib.request

import certifi
import nibabel as nib
import numpy as np

MODEL_HASH = '1f8a7a056d0ebc00486edc43c26768bf1c12eaa6df9dd172e34598003be95eb3'
MODEL_REV = '430f2c82a96dce44b455de5876d43a5a9f753cb2'
DATA_REV = '327dd551a9e51295c34e14f311d8330ef44b6cac'
DATA_REPO = 'Angelou0516/msd-pancreas'
FILES = {
 '001': ('104024672d8ae993905621573425073f76a1fd58d1b67a902e98e849e358f057',
         'c23328f95bc048c3d49c31b8b52edd232a741781c8a56763c94674595736e67d'),
 '004': ('eb16eced003524fa005e28b2822c0b53503f1223d758cdf72528fad359aa10ba',
         'e12664490fe02b8d9d2857b7c66a4176c8445020bf02faf1597395dc98616a50'),
 '005': ('d2cc93d434bb967a2ea69109d32abfaf5eaed350d884fb6a9b098b66fdc48c2d',
         '6297b554d5cb36e150072a01d14ddff9249f821062641530687918c14f054f9a'),
 '006': ('5a9fb6e0e3930ced6377d9233c3633b3eca9d95795858c9a461ccb1b3903de87',
         '743d0015c01dadd3b5a3d5315b37ea9e06088399b9ba4b8e7b9e1ca6299d4025'),
}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url, path, expected, limit):
    if path.exists():
        if sha(path) != expected:
            raise RuntimeError(f'Existing file hash mismatch: {path}')
        return
    part = path.with_suffix(path.suffix + '.incomplete')
    # Exclusive partial creation prevents concurrent preparations from colliding.
    request = urllib.request.Request(url, headers={'User-Agent': 'compression-screen/1'})
    with urllib.request.urlopen(request, timeout=60,
         context=ssl.create_default_context(cafile=certifi.where())) as response, part.open('xb') as output:
        total = 0
        while chunk := response.read(4*1024*1024):
            total += len(chunk)
            if total > limit:
                raise RuntimeError('Download exceeded explicit byte bound')
            output.write(chunk)
    if sha(part) != expected:
        raise RuntimeError(f'Download hash mismatch: {part}')
    os.replace(part, path)


def main():
    root = Path(__file__).resolve().parent
    data = root / 'inputs'; data.mkdir(exist_ok=True)
    if shutil.disk_usage(root).free < 2*1024**3:
        raise RuntimeError('Need 2GiB headroom for isolated probe')
    model = data / 'autoencoder_v1.pt'
    fetch(f'https://huggingface.co/nvidia/NV-Generate-CT/resolve/{MODEL_REV}/models/autoencoder_v1.pt',
          model, MODEL_HASH, 90_000_000)
    records = []
    for case, hashes in FILES.items():
        label = data / f'pancreas_{case}_label.nii.gz'
        fetch(f'https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REV}/labelsTr/pancreas_{case}.nii.gz',
              label, hashes[1], 2_000_000)
        im = nib.load(label); array = np.asarray(im.dataobj)
        if im.header.get_xyzt_units()[0] != 'mm' or not set(np.unique(array)) <= {0, 1, 2}:
            raise RuntimeError('MSD label/units convention failed')
        n = int(np.count_nonzero(array == 2))
        records.append({'case': case, 'tumor_voxels': n,
                        'tumor_mm3': float(n * abs(np.linalg.det(im.affine[:3,:3]))),
                        'label': str(label), 'label_sha256': hashes[1]})
    positives = sorted((r for r in records if r['tumor_voxels'] > 0), key=lambda r: (r['tumor_mm3'], r['case']))
    if not positives:
        raise RuntimeError('No actual label-2-positive input; do not submit GPU job')
    selected = positives[:2]
    for row in selected:
        case = row['case']; image = data / f'pancreas_{case}_image.nii.gz'
        fetch(f'https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REV}/imagesTr/pancreas_{case}.nii.gz',
              image, FILES[case][0], 80_000_000)
        ct, label = nib.load(image), nib.load(row['label'])
        if ct.shape != label.shape or not np.allclose(ct.affine, label.affine, rtol=0, atol=1e-5):
            raise RuntimeError('Image/label geometry mismatch')
        row.update(image=str(image), image_sha256=FILES[case][0], shape=list(ct.shape))
    manifest = {'model': str(model), 'model_sha256': MODEL_HASH,
                'model_revision': MODEL_REV, 'data_revision': DATA_REV,
                'data_repo': DATA_REPO, 'tumor_label': 2,
                'selection': 'two smallest physical tumor burdens among four prespecified cases, not globally smallest',
                'mask_inventory': records, 'cases': selected}
    temporary = root / 'inputs_manifest.incomplete'
    with temporary.open('x') as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    os.replace(temporary, root / 'inputs_manifest.json')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
