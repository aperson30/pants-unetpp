"""
Scans every prediction's .npz probability file and reports which ones (if any) are corrupted --
this can happen if a file was only partially written when a process got killed or crashed mid-save.

Run with:
    python check_npz_integrity.py /Scratch/enl014/PanTS_predictions_test
"""
import sys
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    predictions_dir = Path(sys.argv[1])
    npz_files = sorted(predictions_dir.glob("*.npz"))
    print(f"Checking {len(npz_files)} files...", flush=True)

    bad_files = []
    for i, f in enumerate(npz_files, start=1):
        try:
            data = np.load(str(f))
            _ = data["probabilities"].shape  # actually touch the array, not just open the file
        except Exception as e:
            bad_files.append((f.name, str(e)))
        if i % 20 == 0:
            print(f"  checked {i}/{len(npz_files)}", flush=True)

    print(f"\n{len(bad_files)} corrupted file(s):", flush=True)
    for name, error in bad_files:
        print(f"  {name}: {error}", flush=True)
