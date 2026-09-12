"""
Progress check for the retrain: is the tumor class actually being learned this time?

Why this exists rather than just tailing the log: nnU-Net's per-epoch "Pseudo dice" line reports all
28 classes, and the tumor is the LAST value. Tumor learning here has a delayed onset -- in the run
that worked, tumor pseudo-Dice was 0.0 for the first ~300 epochs, then rose sharply. So a single
recent value tells you almost nothing; what matters is the trend across 100-epoch windows, compared
against the run that is known to have worked.

Reference trajectory from the WORKING run (UNet++, nnUNetPlansBS4, 4 GPUs), % of epochs in each
window where any tumor was seen at all:
    ep    0- 300 :   0%
    ep  300- 400 :   7%
    ep  400- 500 :  58%
    ep  500- 600 :  87%
    ep  600- 700 :  97%
    ep  700-1000 :  98-100%   (mean tumor Dice ~0.38 by the end)

The failed BS2 runs sat at 0-3% the whole way. So: do not panic before epoch 400, and check that by
epoch ~500 the current run is climbing toward the reference column, not flat.

Run with:
    python3 check_tumor_progress.py
"""
import glob
import re
from pathlib import Path

RESULTS = "/Scratch/enl014/nnUNet_results/Dataset001_PanTS"

# name -> results subfolder
RUNS = {
    "unetpp_ds   (retrain)": "nnUNetTrainerUNetPlusPlus__nnUNetPlansBS4__3d_fullres",
    "unetpp_nods (retrain)": "nnUNetTrainerUNetPlusPlusNoDeepSupervision__nnUNetPlansBS4__3d_fullres",
    "default_ds  (retrain)": "nnUNetTrainer__nnUNetPlansBS4__3d_fullres",
    "default_nods(retrain)": "nnUNetTrainerNoDeepSupervision__nnUNetPlansBS4__3d_fullres",
}

REFERENCE = {0: 0, 100: 0, 200: 0, 300: 7, 400: 58, 500: 87, 600: 97, 700: 98, 800: 100, 900: 99}


def tumor_series(folder: str) -> list:
    """Last value of every 'Pseudo dice [...]' line = the tumor class (class 28)."""
    vals = []
    pattern = f"{RESULTS}/{folder}/fold_0/training_log_*.txt"
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for line in f:
                if "Pseudo dice" in line:
                    nums = re.findall(r"np\.float32\(([0-9.eE+-]+)\)", line)
                    if nums:
                        vals.append(float(nums[-1]))
    return vals


if __name__ == "__main__":
    for label, folder in RUNS.items():
        if not Path(f"{RESULTS}/{folder}/fold_0").is_dir():
            print(f"{label}: not started yet")
            continue

        vals = tumor_series(folder)
        if not vals:
            print(f"{label}: started, but no epochs logged yet")
            continue

        print(f"\n=== {label} -- {len(vals)} epochs so far ===")
        print(f"{'epochs':>12}  {'% saw tumor':>12}  {'mean dice':>10}   {'reference':>10}")
        for lo in range(0, len(vals), 100):
            window = vals[lo:lo + 100]
            if not window:
                continue
            nonzero = [v for v in window if v > 0]
            pct = 100 * len(nonzero) / len(window)
            mean = sum(window) / len(window)
            ref = REFERENCE.get(lo)
            ref_str = f"{ref}%" if ref is not None else "-"
            print(f"{lo:5d}-{lo + len(window):5d}  {pct:11.1f}%  {mean:10.4f}   {ref_str:>10}")

        if len(vals) < 400:
            print("  --> too early to judge (the working run showed 0% until epoch ~300)")
        else:
            recent = vals[-100:]
            pct = 100 * len([v for v in recent if v > 0]) / len(recent)
            if pct >= 40:
                print(f"  --> ON TRACK: {pct:.0f}% of the last 100 epochs saw a tumor")
            elif pct >= 10:
                print(f"  --> possibly starting to learn ({pct:.0f}% of last 100 epochs) -- keep watching")
            else:
                print(f"  --> WARNING: only {pct:.0f}% of the last 100 epochs saw a tumor. "
                      f"Past epoch 400 the working run was already at 58%. This run may be "
                      f"repeating the earlier failure.")
