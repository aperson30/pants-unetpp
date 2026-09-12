"""
Re-predicts a trainer's validation-fold cases WITH probability maps saved, then immediately shrinks
each one down to a single number (max tumor-class probability) and deletes the large file -- same
disk-safe batching pattern as predict_and_shrink.py, just scoped to the validation-fold case list
instead of the held-out test set, and sourcing images from imagesTr (since validation-fold cases are
part of the training data's cross-validation split, not the official test set).

Why this exists separately from perform_actual_validation: nnU-Net's own validation step (what
produced the .nii.gz files already sitting in the results folder) was run WITHOUT --npz, so no
probability maps exist yet -- and re-running the whole validation step over would throw away nothing
new (the binary segmentations are already correct), just wastes GPU time. This script only produces
the ONE additional number (max tumor probability per case) needed for AUC, reusing whichever case IDs
already have a predicted .nii.gz in the validation folder, so it stays in sync automatically.

Run with (one call per GPU/trainer):
    CUDA_VISIBLE_DEVICES=0 python3 extract_val_probabilities.py \
        --images-dir /Scratch/enl014/nnUNet_raw/Dataset001_PanTS/imagesTr \
        --validation-dir /Scratch/enl014/nnUNet_results/Dataset001_PanTS/nnUNetTrainer__nnUNetPlansBS2__3d_fullres/fold_0/validation \
        --dataset 1 --config 3d_fullres -tr nnUNetTrainer -p nnUNetPlansBS2 -f 0 \
        --max-probs-csv max_tumor_probs_default_ds.csv \
        --batch-size 20
"""
import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np

TUMOR_CLASS = 28


def extract_and_shrink(output_dir: Path, case_ids: list, csv_writer, csv_file) -> None:
    for case_id in case_ids:
        npz_path = output_dir / f"{case_id}.npz"
        if not npz_path.is_file():
            print(f"  WARNING: no .npz produced for {case_id}, skipping", flush=True)
            continue
        try:
            probabilities = np.load(str(npz_path))["probabilities"]
            max_prob = float(probabilities[TUMOR_CLASS].max())
            csv_writer.writerow([case_id, max_prob])
            csv_file.flush()
            npz_path.unlink()
        except Exception as e:
            print(f"  WARNING: could not shrink {case_id}: {e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True,
                         help="existing validation folder -- its *.nii.gz filenames define the case "
                              "list, and probability output is written into a throwaway subfolder here")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("-tr", required=True)
    parser.add_argument("-p", required=True)
    parser.add_argument("-f", required=True)
    parser.add_argument("--max-probs-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    case_ids = sorted(p.name.removesuffix(".nii.gz") for p in args.validation_dir.glob("*.nii.gz"))
    print(f"{len(case_ids)} validation-fold case(s) to re-predict with probabilities", flush=True)

    output_dir = args.validation_dir.parent / "validation_probs_tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_input_dir = args.validation_dir.parent / "_val_prob_batch_input"

    batches = [case_ids[i:i + args.batch_size] for i in range(0, len(case_ids), args.batch_size)]

    write_header = not args.max_probs_csv.is_file() or args.max_probs_csv.stat().st_size == 0
    already_done = set()
    if not write_header:
        with open(args.max_probs_csv, newline="") as f:
            already_done = {row["case_id"] for row in csv.DictReader(f)}
        print(f"{len(already_done)} case(s) already have a probability recorded, will skip those",
              flush=True)

    with open(args.max_probs_csv, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["case_id", "max_tumor_probability"])
            csv_file.flush()

        for batch_num, batch in enumerate(batches, start=1):
            batch = [c for c in batch if c not in already_done]
            if not batch:
                continue
            print(f"\n=== batch {batch_num}/{len(batches)} ({len(batch)} cases) ===", flush=True)

            tmp_input_dir.mkdir(exist_ok=True)
            for case_id in batch:
                src = args.images_dir / f"{case_id}_0000.nii.gz"
                dst = tmp_input_dir / f"{case_id}_0000.nii.gz"
                if not src.is_file():
                    print(f"  WARNING: source image missing for {case_id}, skipping", flush=True)
                    continue
                if not dst.exists():
                    dst.symlink_to(src.resolve())

            result = subprocess.run([
                "nnUNetv2_predict",
                "-i", str(tmp_input_dir),
                "-o", str(output_dir),
                "-d", args.dataset, "-c", args.config,
                "-tr", args.tr, "-p", args.p, "-f", args.f,
                "--save_probabilities", "--continue_prediction",
            ])
            if result.returncode != 0:
                print(f"  WARNING: batch {batch_num} prediction exited with code "
                      f"{result.returncode} -- shrinking whatever did succeed, continuing", flush=True)

            extract_and_shrink(output_dir, batch, writer, csv_file)
            shutil.rmtree(tmp_input_dir, ignore_errors=True)

    shutil.rmtree(output_dir, ignore_errors=True)
    print(f"\nDone. Max tumor probabilities written to {args.max_probs_csv}", flush=True)
