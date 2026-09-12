"""
Computes the 5 PanTS benchmark metrics (P-Sen, T-Sen, Spe, AUC, DSC) for one trainer's validation
output, matching the definitions used in PanTS's own leaderboard table:

  - DSC:   Dice score on the tumor class ONLY (class 28), averaged over cases that actually HAVE a
           tumor in the ground truth (Dice on an empty-vs-empty mask is undefined/trivially 1.0 and
           not meaningful, so tumor-negative cases are excluded from this average, same convention
           used by segmentation benchmarks generally).
  - P-Sen: patient-wise sensitivity. Among ground-truth-positive cases (has >=1 tumor voxel), the
           fraction where the prediction ALSO has >=1 predicted tumor voxel anywhere in the volume.
           Location doesn't matter here -- just "did we notice something was wrong".
  - T-Sen: tumor-wise sensitivity. Stricter than P-Sen: each ground-truth tumor is a SEPARATE connected
           component (a patient can have multiple tumors). A given true tumor only counts as detected
           if a predicted tumor voxel actually overlaps that specific component -- finding a different,
           unrelated tumor elsewhere in the same patient does not count as detecting this one.
  - Spe:   specificity. Among ground-truth-NEGATIVE cases (no tumor at all), the fraction where the
           prediction correctly predicts zero tumor voxels (no false alarm).
  - AUC:   needs a continuous per-case confidence score (max tumor-class probability), not just the
           binary segmentation -- see extract_val_probabilities.py, which produces the CSV this script
           reads for the AUC calculation. If that CSV isn't available yet, AUC is skipped (reported as
           None) rather than silently computed wrong.

Ground truth convention: labelsTr/<case_id>.nii.gz is the same merged, multi-label volume nnU-Net was
trained on, where class 28 = pancreatic_lesion (matches class_map_abdomenatlas_pants).

Run with:
    python3 compute_tumor_metrics.py \
        --pred-dir /Scratch/enl014/nnUNet_results/Dataset001_PanTS/nnUNetTrainer__nnUNetPlansBS2__3d_fullres/fold_0/validation \
        --labels-dir /Scratch/enl014/nnUNet_raw/Dataset001_PanTS/labelsTr \
        --tumor-class 28 \
        --probs-csv max_tumor_probs_default_ds.csv \
        --out-json metrics_default_ds.json
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label as cc_label

TUMOR_CLASS_DEFAULT = 28


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return np.nan  # both empty -- undefined, caller should exclude this case
    return 2.0 * inter / denom


def tumor_wise_detection(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple:
    """Returns (num_true_tumors, num_detected) for one case, using connected components of the
    ground-truth tumor mask. A true tumor component counts as detected if the prediction has ANY
    voxel overlapping that specific component (voxel-level overlap, not a distance/IoU threshold --
    simple and matches how sensitivity is usually reported for lesion detection benchmarks)."""
    gt_components, n_components = cc_label(gt_mask)
    if n_components == 0:
        return 0, 0
    detected = 0
    for comp_id in range(1, n_components + 1):
        comp_mask = gt_components == comp_id
        if np.logical_and(comp_mask, pred_mask).any():
            detected += 1
    return n_components, detected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--tumor-class", type=int, default=TUMOR_CLASS_DEFAULT)
    parser.add_argument("--probs-csv", type=Path, default=None,
                         help="CSV with columns case_id,max_tumor_probability -- from "
                              "extract_val_probabilities.py. If omitted or missing, AUC is skipped.")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    pred_files = sorted(args.pred_dir.glob("*.nii.gz"))
    print(f"Found {len(pred_files)} predicted cases in {args.pred_dir}", flush=True)

    dice_scores = []
    p_sen_hits, p_sen_total = 0, 0
    spe_hits, spe_total = 0, 0
    t_sen_detected_total, t_sen_true_total = 0, 0
    skipped = []

    for i, pred_path in enumerate(pred_files, start=1):
        case_id = pred_path.name.removesuffix(".nii.gz")
        gt_path = args.labels_dir / f"{case_id}.nii.gz"
        if not gt_path.is_file():
            skipped.append(case_id)
            continue

        pred_vol = nib.load(str(pred_path)).get_fdata(dtype=np.float32)
        gt_vol = nib.load(str(gt_path)).get_fdata(dtype=np.float32)

        pred_mask = pred_vol == args.tumor_class
        gt_mask = gt_vol == args.tumor_class
        gt_has_tumor = gt_mask.any()
        pred_has_tumor = pred_mask.any()

        if gt_has_tumor:
            p_sen_total += 1
            if pred_has_tumor:
                p_sen_hits += 1

            d = dice(pred_mask, gt_mask)
            if not np.isnan(d):
                dice_scores.append(d)

            n_true, n_detected = tumor_wise_detection(pred_mask, gt_mask)
            t_sen_true_total += n_true
            t_sen_detected_total += n_detected
        else:
            spe_total += 1
            if not pred_has_tumor:
                spe_hits += 1

        if i % 200 == 0:
            print(f"  ...{i}/{len(pred_files)} cases processed", flush=True)

    if skipped:
        print(f"WARNING: {len(skipped)} case(s) had no matching ground truth, skipped: "
              f"{skipped[:10]}{'...' if len(skipped) > 10 else ''}", flush=True)

    results = {
        "n_cases_evaluated": len(pred_files) - len(skipped),
        "n_skipped_no_gt": len(skipped),
        "DSC_tumor_mean": float(np.mean(dice_scores)) if dice_scores else None,
        "DSC_tumor_n_positive_cases": len(dice_scores),
        "P_Sen": p_sen_hits / p_sen_total if p_sen_total else None,
        "P_Sen_n_positive_cases": p_sen_total,
        "T_Sen": t_sen_detected_total / t_sen_true_total if t_sen_true_total else None,
        "T_Sen_n_true_tumors": t_sen_true_total,
        "Spe": spe_hits / spe_total if spe_total else None,
        "Spe_n_negative_cases": spe_total,
        "AUC": None,
    }

    if args.probs_csv is not None and args.probs_csv.is_file():
        import csv
        from sklearn.metrics import roc_auc_score

        probs = {}
        with open(args.probs_csv, newline="") as f:
            for row in csv.DictReader(f):
                probs[row["case_id"]] = float(row["max_tumor_probability"])

        y_true, y_score = [], []
        missing = 0
        for pred_path in pred_files:
            case_id = pred_path.name.removesuffix(".nii.gz")
            gt_path = args.labels_dir / f"{case_id}.nii.gz"
            if not gt_path.is_file() or case_id not in probs:
                missing += 1
                continue
            gt_vol = nib.load(str(gt_path)).get_fdata(dtype=np.float32)
            y_true.append(int((gt_vol == args.tumor_class).any()))
            y_score.append(probs[case_id])

        if missing:
            print(f"WARNING: {missing} case(s) missing from probs CSV or GT, excluded from AUC",
                  flush=True)

        if len(set(y_true)) < 2:
            print("WARNING: AUC undefined -- need both tumor-positive and tumor-negative cases",
                  flush=True)
        else:
            results["AUC"] = float(roc_auc_score(y_true, y_score))
            results["AUC_n_cases"] = len(y_true)
    else:
        print(f"No probs CSV found at {args.probs_csv} -- AUC left as None. "
              f"Run extract_val_probabilities.py first if you need it.", flush=True)

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {args.out_json}:", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
