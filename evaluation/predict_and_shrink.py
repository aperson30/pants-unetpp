"""
Runs nnU-Net prediction in small batches instead of all-at-once, immediately shrinking each batch's
large probability files down to a single number per case (and deleting the originals) before moving
to the next batch. This keeps disk usage bounded to roughly one batch's worth of full-size data at
any moment, instead of needing room for the entire dataset's probability files simultaneously --
which is exactly what caused repeated disk-full crashes when we predicted everything at once first
and only cleaned up afterward.

Can still be run multiple times in parallel across GPUs, same as before (set CUDA_VISIBLE_DEVICES
before launching each one, and use --num-parts/--part-id to split the case list between them).

Run with (one call per GPU, each with a different CUDA_VISIBLE_DEVICES and --part-id):
    CUDA_VISIBLE_DEVICES=0 python predict_and_shrink.py \
        --images-dir /Scratch/enl014/nnUNet_raw/Dataset001_PanTS/imagesTs \
        --output-dir /Scratch/enl014/PanTS_predictions_test \
        --dataset 1 --config 3d_fullres -tr nnUNetTrainerUNetPlusPlus -p nnUNetPlansBS4 -f 0 \
        --max-probs-csv max_tumor_probs.csv \
        --batch-size 20 --num-parts 4 --part-id 0
"""
import argparse
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np

TUMOR_CLASS = 28


def extract_and_shrink(output_dir: Path, case_ids: list, csv_writer) -> None:
    """For each case that now has a probability file, pull out the one number we actually need
    (max tumor-channel confidence) and delete the large original."""
    for case_id in case_ids:
        npz_path = output_dir / f"{case_id}.npz"
        if not npz_path.is_file():
            continue  # this case's prediction didn't produce a probability file -- leave it, don't crash
        try:
            probabilities = np.load(str(npz_path))["probabilities"]
            max_prob = float(probabilities[TUMOR_CLASS].max())
            csv_writer.writerow([case_id, max_prob])
            npz_path.unlink()
        except Exception as e:
            print(f"  WARNING: could not shrink {case_id}: {e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("-tr", required=True)
    parser.add_argument("-p", required=True)
    parser.add_argument("-f", required=True)
    parser.add_argument("--max-probs-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument("--part-id", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tmp_input_dir = Path(f"./_batch_input_part{args.part_id}")

    case_ids = sorted(p.name.removesuffix("_0000.nii.gz") for p in args.images_dir.glob("*_0000.nii.gz"))
    if args.num_parts > 1:
        case_ids = case_ids[args.part_id::args.num_parts]
    print(f"Part {args.part_id}/{args.num_parts}: {len(case_ids)} cases, "
          f"in batches of {args.batch_size}", flush=True)

    batches = [case_ids[i:i + args.batch_size] for i in range(0, len(case_ids), args.batch_size)]

    write_header = not args.max_probs_csv.is_file() or args.max_probs_csv.stat().st_size == 0
    with open(args.max_probs_csv, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["case_id", "max_tumor_probability"])
            csv_file.flush()

        for batch_num, batch in enumerate(batches, start=1):
            print(f"\n=== Part {args.part_id}: batch {batch_num}/{len(batches)} "
                  f"({len(batch)} cases) ===", flush=True)

            # symlink just this batch's images into a small temp folder, so nnUNetv2_predict only
            # touches these cases -- no copying, so this costs no extra disk space
            tmp_input_dir.mkdir(exist_ok=True)
            for case_id in batch:
                src = args.images_dir / f"{case_id}_0000.nii.gz"
                dst = tmp_input_dir / f"{case_id}_0000.nii.gz"
                if not dst.exists():
                    dst.symlink_to(src.resolve())

            result = subprocess.run([
                "nnUNetv2_predict",
                "-i", str(tmp_input_dir),
                "-o", str(args.output_dir),
                "-d", args.dataset, "-c", args.config,
                "-tr", args.tr, "-p", args.p, "-f", args.f,
                "--save_probabilities", "--continue_prediction",
            ])
            if result.returncode != 0:
                print(f"  WARNING: batch {batch_num} prediction exited with code "
                      f"{result.returncode} -- shrinking whatever did succeed, continuing", flush=True)

            extract_and_shrink(args.output_dir, batch, writer)
            csv_file.flush()
            shutil.rmtree(tmp_input_dir)

    print(f"\nPart {args.part_id} done. Max probabilities appended to {args.max_probs_csv}", flush=True)
