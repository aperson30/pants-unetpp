"""Bounded single public-case transport/geometry check, not cohort selection.

Run with bundled Python. Uses ZIP CRC verification, pinned public manual label,
exact IDs, no GPU, no crop and no image publishing. Leaves failed .partial files
for diagnosis; success marker only after full finite/geometry checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile
import zlib

import nibabel as nib
import numpy as np

from list_remote_zip import RangeReader


def fetch(url, limit):
    with urllib.request.urlopen(url, timeout=20) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Manual label exceeds download bound')
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eligibility', type=Path, required=True)
    parser.add_argument('--study', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-member-mb', type=int, choices=(30, 60), default=30,
                        help='Explicit transport cap; 60 is only for locked replication cases')
    parser.add_argument('--verify-existing', action='store_true',
                        help='Verify a completed transfer against fresh ZIP CRC; never overwrite inputs')
    args = parser.parse_args()
    audit = json.loads(args.eligibility.read_text())
    selected = [c for c in audit['candidates'] if c['study'] == args.study]
    if len(selected) != 1:
        raise ValueError('Require one audited provisional candidate')
    case = selected[0]
    expected = case['archive']
    if expected['compression'] != 0 or expected['encrypted'] or expected['compressed'] > args.max_member_mb*1_000_000:
        raise ValueError('Require stored, unencrypted member within explicit transport cap')
    if args.verify_existing:
        if not args.output.is_dir() or (args.output/'completion.json').exists():
            raise ValueError('Require an incomplete existing transport directory')
    else:
        args.output.mkdir(exist_ok=False)
    if shutil.disk_usage(args.output).free < 1_000_000_000:
        raise ValueError('Insufficient local working space')
    reader = RangeReader('https://zenodo.org/records/10998332/files/batch_1.zip?download=1',
                         byte_limit=(args.max_member_mb+10)*1024**2, max_calls=128,
                         total_timeout=300)
    partial = args.output/'image.nii.gz.partial'
    digest = hashlib.sha256()
    with zipfile.ZipFile(reader) as archive:
        member = archive.getinfo(expected['name'])
        if (member.file_size, member.compress_size, member.compress_type, member.header_offset) != (
                expected['uncompressed'], expected['compressed'], expected['compression'], expected['header_offset']):
            raise ValueError('Archive metadata differs from audited snapshot')
        if member.flag_bits & 1:
            raise ValueError('Encrypted member')
        written = 0
        if args.verify_existing:
            crc = 0
            with (args.output/'image.nii.gz').open('rb') as source:
                while chunk := source.read(1024**2):
                    written += len(chunk)
                    digest.update(chunk)
                    crc = zlib.crc32(chunk, crc)
            if crc != member.CRC:
                raise ValueError('Existing transfer CRC mismatch')
        else:
            with archive.open(member) as source, partial.open('xb') as destination:
                while True:
                    chunk = source.read(1024**2)  # ZipExtFile verifies CRC at EOF.
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > member.file_size:
                        raise ValueError('Member size exceeded')
                    digest.update(chunk)
                    destination.write(chunk)
        if written != member.file_size:
            raise ValueError('Incomplete member')
    image_path = args.output/'image.nii.gz'
    if not args.verify_existing:
        partial.rename(image_path)
    sha = audit['summary']['provenance']['panorama_labels']
    label = fetch(f'https://raw.githubusercontent.com/DIAGNijmegen/panorama_labels/{sha}/manual_labels/{args.study}.nii.gz',
                  limit=5_000_000)
    label_path = args.output/'manual_label.nii.gz'
    if args.verify_existing:
        if label_path.read_bytes() != label:
            raise ValueError('Existing mask differs from pinned source')
    else:
        with label_path.open('xb') as destination:
            destination.write(label)
    image, mask = nib.load(image_path), nib.load(label_path)
    if len(image.shape) != 3 or np.prod(image.shape) > 100_000_000:
        raise ValueError('Unexpected/oversized NIfTI shape')
    if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, atol=1e-4, rtol=0):
        raise ValueError('CT/manual-label geometry mismatch')
    data, labels = image.get_fdata(dtype=np.float32), mask.get_fdata(dtype=np.float32)
    if not np.isfinite(data).all() or not np.isfinite(labels).all() or not np.isfinite(image.affine).all():
        raise ValueError('Nonfinite data/geometry')
    if not set(np.unique(labels)).issubset(set(range(7))) or not (labels == 1).any():
        raise ValueError('Unexpected/empty manual PDAC label')
    voxel_ml = abs(np.linalg.det(image.affine[:3, :3]))/1000
    if image.header.get_xyzt_units()[0] != 'mm' or voxel_ml <= 0:
        raise ValueError('Require millimeter spatial units and nonsingular affine')
    result = dict(completed=True, purpose='transport/geometry ONLY, not cohort selection or clinical endpoint',
                  case=case, label_commit=sha, ct_sha256=digest.hexdigest(),
                  label_sha256=hashlib.sha256(label).hexdigest(), zip_crc32=member.CRC,
                  archive_etag=reader.etag, archive_bytes=reader.total,
                  fetched_archive_bytes=reader.used, range_requests=reader.calls,
                  shape=list(image.shape), spacing_mm=[float(z) for z in image.header.get_zooms()],
                  nifti_spatial_units=image.header.get_xyzt_units()[0],
                  hu_min=float(data.min()), hu_max=float(data.max()),
                  tumor_voxels=int((labels == 1).sum()), tumor_ml=float((labels == 1).sum()*voxel_ml))
    (args.output/'completion.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
