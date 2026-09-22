"""Real-case equivalence check for the parallel PanTS converter path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--parallel", type=Path, required=True)
    args = parser.parse_args()

    serial_json = json.loads((args.serial / "dataset.json").read_text(encoding="utf-8"))
    parallel_json = json.loads((args.parallel / "dataset.json").read_text(encoding="utf-8"))
    if serial_json != parallel_json:
        raise RuntimeError("dataset.json differs between serial and parallel conversion")

    serial_images = sorted((args.serial / "imagesTr").glob("*.nii.gz"))
    parallel_images = sorted((args.parallel / "imagesTr").glob("*.nii.gz"))
    if [p.name for p in serial_images] != [p.name for p in parallel_images]:
        raise RuntimeError("image case lists differ")
    for serial_path, parallel_path in zip(serial_images, parallel_images):
        if serial_path.read_bytes() != parallel_path.read_bytes():
            raise RuntimeError(f"copied CT differs byte-for-byte: {serial_path.name}")

    serial_labels = sorted((args.serial / "labelsTr").glob("*.nii.gz"))
    parallel_labels = sorted((args.parallel / "labelsTr").glob("*.nii.gz"))
    if [p.name for p in serial_labels] != [p.name for p in parallel_labels]:
        raise RuntimeError("label case lists differ")
    for serial_path, parallel_path in zip(serial_labels, parallel_labels):
        serial_image = nib.load(str(serial_path))
        parallel_image = nib.load(str(parallel_path))
        if not np.array_equal(np.asanyarray(serial_image.dataobj), np.asanyarray(parallel_image.dataobj)):
            raise RuntimeError(f"label voxels differ: {serial_path.name}")
        if not np.array_equal(serial_image.affine, parallel_image.affine):
            raise RuntimeError(f"label affine differs: {serial_path.name}")
        if serial_image.header.binaryblock != parallel_image.header.binaryblock:
            raise RuntimeError(f"label header differs: {serial_path.name}")

    print(f"PARALLEL_CONVERSION_PARITY_PASS cases={len(serial_images)}", flush=True)


if __name__ == "__main__":
    main()
