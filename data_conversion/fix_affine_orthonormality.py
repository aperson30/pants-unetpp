"""
Finds every image/label file whose orientation directions aren't perfectly orthonormal (see
check_affine_orthonormality.py) and corrects them in place, nudging each to the mathematically
*nearest* perfectly-square version -- appropriate only when the deviation is tiny (floating-point
rounding noise), NOT for genuinely tilted scans, since this doesn't attempt to preserve any real
physical skew.

Run on the server with:
    python fix_affine_orthonormality.py /Scratch/enl014/nnUNet_raw/Dataset001_PanTS
    python fix_affine_orthonormality.py /Scratch/enl014/PanTS_test_answer_key
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

THRESHOLD = 1e-4


def orthonormality_deviation(affine: np.ndarray) -> float:
    rotation_part = affine[:3, :3]
    norms = np.linalg.norm(rotation_part, axis=0)
    normalized = rotation_part / norms
    should_be_identity = normalized.T @ normalized
    return float(np.max(np.abs(should_be_identity - np.eye(3))))


def nearest_orthonormal_affine(affine: np.ndarray) -> np.ndarray:
    """Nudges the direction part of the affine to the closest perfectly-square version,
    keeping voxel spacing and position unchanged."""
    rotation_part = affine[:3, :3]
    spacing = np.linalg.norm(rotation_part, axis=0)
    directions = rotation_part / spacing
    U, _, Vt = np.linalg.svd(directions)
    fixed_directions = U @ Vt
    fixed = affine.copy()
    fixed[:3, :3] = fixed_directions * spacing
    return fixed


def fix_folder(folder: Path) -> int:
    fixed_count = 0
    for f in sorted(folder.glob("*.nii.gz")):
        img = nib.load(str(f))
        dev = orthonormality_deviation(img.affine)
        if dev > THRESHOLD:
            data = np.asanyarray(img.dataobj)
            fixed_affine = nearest_orthonormal_affine(img.affine)
            nib.save(nib.Nifti1Image(data, fixed_affine, img.header), str(f))
            print(f"  fixed {f.name} (deviation was {dev:.6f})")
            fixed_count += 1
    return fixed_count


if __name__ == "__main__":
    target_dir = Path(sys.argv[1])
    total_fixed = 0
    # handles either a full nnU-Net dataset dir (with imagesTr/labelsTr/imagesTs subfolders)
    # or a flat folder of label files (like the separate test-answer-key folder)
    subfolders = [target_dir / s for s in ["imagesTr", "labelsTr", "imagesTs"] if (target_dir / s).is_dir()]
    if not subfolders:
        subfolders = [target_dir]

    for folder in subfolders:
        print(f"--- fixing {folder} ---")
        total_fixed += fix_folder(folder)

    print(f"\nTotal files fixed: {total_fixed}")
