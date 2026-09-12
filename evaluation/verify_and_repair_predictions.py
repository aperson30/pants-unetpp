"""
Comprehensive audit of the prediction output folder: for every expected test case, checks that
ALL THREE required files exist (.nii.gz segmentation, .npz probabilities, .pkl metadata) AND that
the readable ones actually load without error (catches truncated/corrupted files, not just missing
ones). Then automatically repairs anything broken by clearing it out and re-predicting just that
one case, one at a time -- and re-checks itself afterward to make sure the repair actually worked.

This exists because we hit the same underlying problem (nnU-Net silently writing some but not all
of a case's output files, no error shown) multiple times in different disguises -- 8 missing files,
then 346 corrupted, then 1 more missing -- always discovered by accident. This runs the same
thorough check every time instead of relying on noticing a discrepancy.

Run with (always under nohup):
    nohup python -u verify_and_repair_predictions.py \
        --images-dir /Scratch/enl014/nnUNet_raw/Dataset001_PanTS/imagesTs \
        --output-dir /Scratch/enl014/PanTS_predictions_test \
        --dataset 1 --config 3d_fullres -tr nnUNetTrainerUNetPlusPlus -p nnUNetPlansBS4 -f 0 \
        > verify_repair.log 2>&1 &
"""
import argparse
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def diagnose(case_ids: list, output_dir: Path, already_extracted: set) -> dict:
    """Returns {case_id: reason} for every case with a real problem.
    already_extracted: case IDs already recorded in max_tumor_probs.csv -- their .npz was
    deliberately deleted on purpose after extraction, so a missing .npz for these is fine, not
    a problem to repair."""
    problems = {}
    for case_id in case_ids:
        nii_path = output_dir / f"{case_id}.nii.gz"
        npz_path = output_dir / f"{case_id}.npz"
        pkl_path = output_dir / f"{case_id}.pkl"

        if not nii_path.is_file():
            problems[case_id] = "missing .nii.gz"
            continue
        if not pkl_path.is_file():
            problems[case_id] = "missing .pkl"
            continue
        if case_id not in already_extracted and not npz_path.is_file():
            problems[case_id] = "missing .npz (and not yet extracted)"
            continue

        try:
            nib.load(str(nii_path)).get_fdata(dtype=np.float32)
        except Exception as e:
            problems[case_id] = f"corrupted .nii.gz: {e}"
            continue
        if npz_path.is_file():
            try:
                data = np.load(str(npz_path))
                _ = data["probabilities"].shape
            except Exception as e:
                problems[case_id] = f"corrupted .npz: {e}"
                continue

    return problems


def repair_one_case(case_id: str, images_dir: Path, output_dir: Path, args) -> None:
    """Clears out whatever partial/broken files exist for this case, then predicts it fresh."""
    for suffix in [".nii.gz", ".npz", ".pkl"]:
        f = output_dir / f"{case_id}{suffix}"
        if f.is_file():
            f.unlink()

    tmp_dir = Path(f"/tmp/repair_{case_id}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    src = images_dir / f"{case_id}_0000.nii.gz"
    (tmp_dir / f"{case_id}_0000.nii.gz").symlink_to(src.resolve())

    subprocess.run([
        "nnUNetv2_predict",
        "-i", str(tmp_dir), "-o", str(output_dir),
        "-d", args.dataset, "-c", args.config,
        "-tr", args.tr, "-p", args.p, "-f", args.f,
        "--save_probabilities", "--continue_prediction",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("-tr", required=True)
    parser.add_argument("-p", required=True)
    parser.add_argument("-f", required=True)
    parser.add_argument("--max-repair-rounds", type=int, default=3)
    parser.add_argument("--max-probs-csv", type=Path, default=None,
                         help="if given, case IDs already recorded here are treated as having a "
                              "valid (deliberately deleted) .npz, not a missing one")
    args = parser.parse_args()

    already_extracted = set()
    if args.max_probs_csv is not None and args.max_probs_csv.is_file():
        import csv
        with open(args.max_probs_csv, newline="") as f:
            already_extracted = {row["case_id"] for row in csv.DictReader(f)}
        print(f"{len(already_extracted)} case(s) already extracted (their .npz being gone is expected).",
              flush=True)

    case_ids = sorted(p.name.removesuffix("_0000.nii.gz") for p in args.images_dir.glob("*_0000.nii.gz"))
    print(f"Auditing {len(case_ids)} expected cases...", flush=True)

    for round_num in range(1, args.max_repair_rounds + 1):
        problems = diagnose(case_ids, args.output_dir, already_extracted)
        if not problems:
            print(f"\nRound {round_num}: all {len(case_ids)} cases verified clean. Done.", flush=True)
            sys.exit(0)

        print(f"\nRound {round_num}: found {len(problems)} problem case(s):", flush=True)
        for case_id, reason in problems.items():
            print(f"  {case_id}: {reason}", flush=True)

        for i, case_id in enumerate(problems, start=1):
            print(f"  Repairing [{i}/{len(problems)}]: {case_id}", flush=True)
            repair_one_case(case_id, args.images_dir, args.output_dir, args)

    # one final check after the last repair round, to report honestly if anything is still broken
    remaining = diagnose(case_ids, args.output_dir, already_extracted)
    if remaining:
        print(f"\nWARNING: {len(remaining)} case(s) still broken after {args.max_repair_rounds} "
              f"repair rounds -- needs manual investigation: {remaining}", flush=True)
        sys.exit(1)
    else:
        print(f"\nAll {len(case_ids)} cases verified clean after repairs. Done.", flush=True)
