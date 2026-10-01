"""Publication helpers. No inference, geometry or metric conventions changed."""
from __future__ import annotations
import argparse
import os
import shutil
import time
from pathlib import Path
import nibabel as nib
import numpy as np
from evaluation.predict_and_shrink import file_hash


def retire_report(path: Path) -> None:
    """Keep previous reports as history, never as a current success marker."""
    if path.exists():
        path.replace(path.with_name(f'{path.name}.history.{time.time_ns()}'))


def publish_ground_truth(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob('*.nii.gz')):
        target = destination / path.name
        expected = file_hash(path)
        if target.exists():
            if file_hash(target) == expected:
                continue
            # Repair only an unreadable partial copy. A readable but different
            # GT is a provenance change, not permission to overwrite evidence.
            try:
                data = np.asanyarray(nib.load(target).dataobj)
            except Exception:
                pass
            else:
                del data
                raise RuntimeError(f'changed readable ground truth: {target}')
        temporary = target.with_name(target.name + '.copy.tmp')
        shutil.copyfile(path, temporary)
        if file_hash(temporary) != expected:
            raise RuntimeError(f'ground-truth copy checksum mismatch: {path}')
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        temporary.replace(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    publish_ground_truth(args.source, args.destination)
