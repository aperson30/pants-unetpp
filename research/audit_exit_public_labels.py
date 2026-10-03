"""Bounded CPU-only label audit for the fixed development-validation screen.

No CT download, model loading, scheduler command or production modification.
Inputs live outside Git in a new exclusive directory. A failure is preserved.
"""
import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

REVISION = '0254c148f3c05fd3913e42d16982b010a048167c'
BASE = 'https://huggingface.co'
REPO = 'BodyMaps/iPanTSMini'
MAX_TOTAL = 12 * 1024 * 1024
MAX_FILE = 3 * 1024 * 1024
MAX_VOXELS = 200_000_000


def request(url):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={'User-Agent': 'pants-exit-label-audit/1'}), timeout=20)


def metadata(case):
    paths = [f'mask_only/{case}', f'mask_only/{case}/segmentations']
    found = {}
    for path in paths:
        with request(f'{BASE}/api/datasets/{REPO}/tree/{REVISION}/{path}') as r:
            payload = r.read(1_000_001)
        if len(payload) > 1_000_000:
            raise RuntimeError('Metadata response exceeded bound')
        for entry in json.loads(payload):
            if entry.get('type') == 'file':
                found[entry['path']] = entry
    return found


def fetch(entry, target, remaining, max_file=MAX_FILE):
    size = entry['size']
    expected = entry['lfs']['oid']
    if not 0 < size <= min(max_file, remaining) or len(expected) != 64:
        raise RuntimeError('File size/hash gate failed')
    digest = hashlib.sha256()
    count = 0
    with request(f'{BASE}/datasets/{REPO}/resolve/{REVISION}/{entry["path"]}') as r:
        with target.open('xb') as out:
            while True:
                block = r.read(min(65536, size - count + 1))
                if not block:
                    break
                count += len(block)
                if count > size:
                    raise RuntimeError('Download exceeded declared byte size')
                digest.update(block)
                out.write(block)
    if count != size or digest.hexdigest() != expected:
        raise RuntimeError('Download size/hash mismatch')
    return count, expected


def read_volume(path):
    img = nib.load(str(path))
    if len(img.shape) != 3 or math.prod(img.shape) > MAX_VOXELS:
        raise RuntimeError('Volume shape/size gate failed')
    if not np.isfinite(img.affine).all():
        raise RuntimeError('Nonfinite affine')
    data = np.asanyarray(img.dataobj)
    if not np.isfinite(data).all():
        raise RuntimeError('Nonfinite label values')
    return img, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases = manifest['cases']
    if len(cases) != 6 or len({r['case_id'] for r in cases}) != 6:
        raise RuntimeError('Requires the fixed six-case manifest')
    args.root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    total = 0
    rows = []
    for row in cases:
        case = row['case_id']
        if not (case.startswith('PanTS_') and len(case) == 14 and case[6:].isdigit()):
            raise RuntimeError('Invalid case ID')
        entries = metadata(case)
        files = []
        for relative, suffix in [('combined_labels.nii.gz', 'combined'),
                                 ('segmentations/pancreatic_lesion.nii.gz', 'lesion')]:
            key = f'mask_only/{case}/{relative}'
            target = args.root / f'{case}_{suffix}.nii.gz'
            count, sha = fetch(entries[key], target, MAX_TOTAL-total)
            total += count
            files.append({'path': str(target), 'source': key, 'sha256': sha,
                          'bytes': count})
        combined_img, combined = read_volume(Path(files[0]['path']))
        lesion_img, lesion = read_volume(Path(files[1]['path']))
        if (combined.shape != lesion.shape or
                not np.allclose(combined_img.affine, lesion_img.affine, rtol=0, atol=1e-6)):
            raise RuntimeError('Combined/individual label geometry mismatch')
        if np.any(combined < 0) or np.any(combined > 28) or np.any(combined != np.floor(combined)):
            raise RuntimeError('Combined labels outside integer0..28')
        tumor = combined == 28
        if not np.array_equal(tumor, lesion > 0):
            raise RuntimeError('Combined class28 differs from individual lesion mask')
        actual = int(np.count_nonzero(tumor))
        if actual != row['n_ref_voxels']:
            raise RuntimeError(f'{case}: public mask count differs from saved validation reference')
        result = {'case_id': case, 'n_ref_voxels': actual, 'shape': list(combined.shape),
                  'affine': combined_img.affine.tolist(), 'files': files,
                  'class28_matches_individual_lesion': True}
        rows.append(result)
        print(json.dumps({'case_id': case, 'n_ref_voxels': actual, 'audit': 'PASS'}), flush=True)
        del combined, lesion, tumor
    report = {'revision': REVISION, 'dataset': REPO, 'bytes_downloaded': total,
              'elapsed_seconds': time.monotonic()-start, 'cases': rows,
              'scope': 'Label/count/geometry audit only; not original archive byte identity, CT verification, model response or clinical evidence'}
    (args.root/'label_audit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
