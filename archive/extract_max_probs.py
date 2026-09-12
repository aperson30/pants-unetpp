"""
Our scoring script only ever needs ONE number out of each large .npz probability file: the highest
tumor-channel value anywhere in the scan (used for AUC). The other 28 channels (background + 27
unrelated organs) are never touched. This script extracts that one number per case into a small CSV,
then deletes the original large .npz files -- freeing the vast majority of their disk space, since
each file only needed to hand over a single float.

IMPORTANT: writes each row to the CSV and flushes it to disk IMMEDIATELY, before deleting that same
file's original -- so an interruption (lost connection, crash) can only ever lose the ONE file
currently in progress, never anything already processed. (An earlier version of this script wrote
the CSV only once at the very end, which caused real data loss when a connection dropped partway
through -- don't reintroduce that.)

Run with (always under nohup, so a dropped connection can't kill it mid-run):
    nohup python -u extract_max_probs.py /Scratch/enl014/PanTS_predictions_test max_tumor_probs.csv > extract.log 2>&1 &
"""
import csv
import sys
from pathlib import Path

import numpy as np

TUMOR_CLASS = 28

if __name__ == "__main__":
    predictions_dir = Path(sys.argv[1])
    out_csv = Path(sys.argv[2])

    npz_files = sorted(predictions_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} probability files.", flush=True)

    # if this is a resumed run after an interruption, don't redo (or lose) rows already saved
    already_done = set()
    if out_csv.is_file():
        with open(out_csv, newline="") as f:
            already_done = {row["case_id"] for row in csv.DictReader(f)}
        print(f"Resuming: {len(already_done)} case(s) already extracted previously.", flush=True)

    write_header = not out_csv.is_file()
    bad = []
    extracted_count = 0
    with open(out_csv, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["case_id", "max_tumor_probability"])
            csv_file.flush()

        for i, f in enumerate(npz_files, start=1):
            case_id = f.name.removesuffix(".npz")
            if case_id in already_done:
                continue
            try:
                probabilities = np.load(str(f))["probabilities"]
                max_prob = float(probabilities[TUMOR_CLASS].max())
            except Exception as e:
                print(f"  SKIPPING {case_id}, could not read: {e}", flush=True)
                bad.append(case_id)
                continue

            # write + flush to disk BEFORE deleting -- the value is safely saved before the
            # original file is removed, so interruption can never lose already-processed data
            writer.writerow([case_id, max_prob])
            csv_file.flush()
            f.unlink()
            extracted_count += 1
            if i % 20 == 0:
                print(f"  [{i}/{len(npz_files)}] extracted and freed: {case_id}", flush=True)

    print(f"\nDone. Extracted {extracted_count} new value(s) to {out_csv}.", flush=True)
    if bad:
        print(f"WARNING: {len(bad)} file(s) could not be read (left in place, not deleted): {bad}", flush=True)
