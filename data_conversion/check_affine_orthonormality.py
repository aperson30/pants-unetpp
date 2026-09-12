"""
Scans every converted image/label file and checks whether its orientation directions are
"orthonormal" (perfectly at right angles to each other) -- the property SimpleITK (nnU-Net's file
reader) requires but nibabel doesn't strictly enforce, which is why our earlier 3-case sanity check
(done with nibabel) didn't catch this.

Run on the server with:
    python check_affine_orthonormality.py /Scratch/enl014/nnUNet_raw/Dataset001_PanTS
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def orthonormality_deviation(affine: np.ndarray) -> float:
    """0.0 = perfectly orthonormal. Bigger numbers = more skewed."""
    rotation_part = affine[:3, :3]
    norms = np.linalg.norm(rotation_part, axis=0)
    normalized = rotation_part / norms
    should_be_identity = normalized.T @ normalized
    return float(np.max(np.abs(should_be_identity - np.eye(3))))


def check_folder(folder: Path, threshold: float = 1e-4) -> list:
    bad = []
    files = sorted(folder.glob("*.nii.gz"))
    for i, f in enumerate(files, start=1):
        try:
            img = nib.load(str(f))
            dev = orthonormality_deviation(img.affine)
            if dev > threshold:
                bad.append((f.name, dev))
        except Exception as e:
            bad.append((f.name, f"FAILED TO LOAD: {e}"))
        if i % 1000 == 0:
            print(f"  checked {i}/{len(files)}")
    return bad


if __name__ == "__main__":
    dataset_dir = Path(sys.argv[1])
    for sub in ["imagesTr", "labelsTr", "imagesTs"]:
        folder = dataset_dir / sub
        if not folder.is_dir():
            continue
        print(f"--- checking {sub} ---")
        bad = check_folder(folder)
        print(f"{len(bad)} problematic file(s) in {sub}")
        for name, dev in bad[:30]:
            print(f"  {name}: {dev}")
        print()
