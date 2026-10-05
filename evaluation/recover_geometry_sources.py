"""Recover six flagged source cases; never modify evaluation evidence.

Archive SHA256 must match the actual original job before any member is copied.
No tar extraction API, source code/checkpoint execution, repairs or metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.request
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np

CASES = {f'PanTS_{i:08d}' for i in (9232, 9357, 9362, 9452, 9480, 9515)}
ARCHIVES = (
    ('PanTSMini_ImageTe_00009001_00009901.tar.gz',
     'https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/3b1cd61108116b58ea5c1ddb3512c1847d965f96/PanTSMini_ImageTe_00009001_00009901.tar.gz?download=true',
     'a7930a2ed1d5638f334ac6614a30b1474832c7dbf24d228a19c5ecba7fe9d7aa', 27994666473),
    ('PanTSMini_Label.tar.gz', 'https://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz',
     '2de0c363c73c8106b0456d49f3e5057641fefb5383a03f6cf13b4f300a672c83', 15561944549),
)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def download(root, name, url, digest, size):
    target = root / name
    if target.exists():
        if target.stat().st_size != size or sha(target) != digest:
            raise RuntimeError('existing archive differs: ' + name)
        return target
    part = root / (name + '.part')
    # Never truncate a failed previous attempt or guess that it is resumable.
    with urllib.request.urlopen(url, timeout=120) as response, part.open('xb') as stream:
        declared = response.headers.get('Content-Length')
        if declared is not None and int(declared) != size:
            raise RuntimeError('advertised source size changed: ' + name)
        h = hashlib.sha256(); count = 0; next_report = 1024**3
        while True:
            block = response.read(8 * 1024**2)
            if not block:
                break
            count += len(block)
            if count > size or shutil.disk_usage(root).free < 5 * 1024**3:
                raise RuntimeError('download size/disk bound reached')
            stream.write(block); h.update(block)
            if count >= next_report:
                print(f'{name}: {count}/{size} bytes', flush=True)
                next_report += 1024**3
        stream.flush(); os.fsync(stream.fileno())
    if count != size or h.hexdigest() != digest:
        raise RuntimeError('original source checksum mismatch: ' + name)
    part.replace(target)
    return target


def recover(archive, destination):
    inventory = []
    copied_bytes = 0
    with tarfile.open(archive, 'r|gz') as source:
        for member in source:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                raise RuntimeError('unsafe archive member')
            if not member.isfile() or not member.name.endswith('.nii.gz'):
                continue
            matches = CASES.intersection(name.parts)
            if not matches:
                continue
            if len(matches) != 1 or member.size > 1024**3:
                raise RuntimeError('ambiguous/oversized source member')
            copied_bytes += member.size
            if copied_bytes > 8 * 1024**3:
                raise RuntimeError('selected extraction size bound')
            target = destination.joinpath(*name.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 8 * 1024**2)
            if target.stat().st_size != member.size:
                raise RuntimeError('short source-member copy')
            inventory.append({'member': member.name, 'bytes': member.size, 'sha256': sha(target)})
    return inventory


def header(path):
    image = nib.load(str(path))
    return {'shape': list(image.shape), 'affine': image.affine.tolist(),
            'axis_codes': list(nib.aff2axcodes(image.affine)),
            'qform_code': int(image.header['qform_code']),
            'sform_code': int(image.header['sform_code'])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--evaluation-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.destination
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 60 * 1024**3:
        raise RuntimeError('need 60 GiB staging headroom')
    provenance = json.loads((args.evaluation_dir / 'unetpp_ds/prediction_provenance.json').read_text())
    all_inventory = {}
    for name, url, digest, size in ARCHIVES:
        archive = download(root, name, url, digest, size)
        destination = root / ('images' if 'ImageTe' in name else 'labels')
        all_inventory[name] = recover(archive, destination)
        print(f'{name}: selected {len(all_inventory[name])} source members', flush=True)
    cases = {}
    for cid in sorted(CASES):
        cts = list((root / 'images').glob(f'**/{cid}/ct.nii.gz'))
        if len(cts) != 1:
            raise RuntimeError(f'{cid}: expected exactly one original CT, found {len(cts)}')
        ct = cts[0]
        if sha(ct) != provenance['inputs'][cid]:
            raise RuntimeError(f'{cid}: original CT differs from prediction input; inspect conversion history')
        masks = sorted((root / 'labels').glob(f'**/{cid}/**/*.nii.gz'))
        if not masks:
            raise RuntimeError(f'{cid}: no individual source masks')
        ct_header = header(ct)
        organs = {}
        for mask in masks:
            value = header(mask)
            value['same_shape_as_ct'] = value['shape'] == ct_header['shape']
            value['max_affine_delta_from_ct'] = float(np.abs(
                np.array(value['affine']) - np.array(ct_header['affine'])).max())
            value['sha256'] = sha(mask)
            if mask.name in organs:
                raise RuntimeError('duplicate source organ name')
            organs[mask.name] = value
        cases[cid] = {'ct': ct_header, 'ct_sha256': sha(ct), 'organs': organs}
        print(f'{cid}: CT hash matches prediction input; {len(organs)} organ headers recovered', flush=True)
    report = {'cases': cases, 'archive_inventory': all_inventory,
              'scope': 'source recovery and headers only; voxel mapping NOT certified'}
    with (root / 'source_geometry_inventory.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('SOURCE_RECOVERY_HASHES_AND_HEADERS_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
