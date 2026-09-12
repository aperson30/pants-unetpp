"""
Finds every .npz probability file that's corrupted (from the earlier disk-full crash) and deletes
that case's ENTIRE output (segmentation .nii.gz, probabilities .npz, and metadata .pkl) so a
subsequent --continue_prediction run will properly redo it -- rather than wrongly treating a
corrupted-but-present file as "already done" just because a file with the right name exists.

Run with:
    python clean_corrupted_predictions.py /Scratch/enl014/PanTS_predictions_test
"""
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    predictions_dir = Path(sys.argv[1])
    npz_files = sorted(predictions_dir.glob("*.npz"))
    print(f"Checking {len(npz_files)} files...")

    corrupted_case_ids = []
    for i, f in enumerate(npz_files, start=1):
        case_id = f.name.removesuffix(".npz")
        try:
            data = np.load(str(f))
            _ = data["probabilities"].shape
        except Exception:
            corrupted_case_ids.append(case_id)
        if i % 100 == 0:
            print(f"  checked {i}/{len(npz_files)}")

    print(f"\n{len(corrupted_case_ids)} corrupted case(s) found. Deleting their output files...")
    for case_id in corrupted_case_ids:
        for suffix in [".nii.gz", ".npz", ".pkl"]:
            file_path = predictions_dir / f"{case_id}{suffix}"
            if file_path.exists():
                file_path.unlink()

    print(f"Deleted output for {len(corrupted_case_ids)} cases.")
    print("Rerun prediction with --continue_prediction to regenerate exactly these cases.")
