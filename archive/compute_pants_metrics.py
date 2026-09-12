"""
Phase 7: compares our UNet++ model's predictions against the real answer key for all 901 official
PanTS test cases, and computes the five metrics the professor asked for: P-Sen, T-Sen, Spe, AUC, DSC.

Run on the server with:
    python compute_pants_metrics.py \
        --predictions-dir /Scratch/enl014/PanTS_predictions_test \
        --ground-truth-dir /Scratch/enl014/PanTS_test_answer_key
"""
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score

TUMOR_CLASS = 28  # pancreatic_lesion, from PanTS's class_map_abdomenatlas_pants


def load_tumor_mask(nifti_path: Path) -> np.ndarray:
    """Load a combined-label file and return just the tumor voxels as True/False."""
    img = nib.load(str(nifti_path))
    labels = np.asanyarray(img.dataobj)
    return labels == TUMOR_CLASS


def get_max_tumor_probability(npz_path: Path) -> float:
    """The model's single highest confidence, anywhere in the scan, that a tumor is present."""
    probabilities = np.load(str(npz_path))["probabilities"]
    return float(probabilities[TUMOR_CLASS].max())


def dice_score(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Standard overlap score: 2x shared voxels, divided by the total voxels in both masks."""
    intersection = np.count_nonzero(gt_mask & pred_mask)
    total = np.count_nonzero(gt_mask) + np.count_nonzero(pred_mask)
    if total == 0:
        return 1.0  # both empty -- not expected here, since we only call this when gt has a tumor
    return 2.0 * intersection / total


def count_tumor_instances_and_detections(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple:
    """Splits the ground-truth tumor voxels into separate individual tumors (connected components),
    and checks each one separately for whether the prediction overlaps it at all."""
    labeled_components, num_instances = ndimage.label(gt_mask, structure=np.ones((3, 3, 3)))
    num_detected = 0
    for instance_id in range(1, num_instances + 1):
        this_tumor = labeled_components == instance_id
        if np.any(this_tumor & pred_mask):
            num_detected += 1
    return num_instances, num_detected


def evaluate_case(case_id: str, predictions_dir: Path, ground_truth_dir: Path,
                   max_probs_lookup: dict = None) -> dict:
    """Everything we need from one single case, gathered in one place.
    max_probs_lookup: if given (case_id -> max probability), use that instead of loading the large
    .npz file directly -- lets us work from extract_max_probs.py's small CSV once the big files have
    been trimmed down and deleted to save disk space."""
    gt_mask = load_tumor_mask(ground_truth_dir / f"{case_id}.nii.gz")
    pred_mask = load_tumor_mask(predictions_dir / f"{case_id}.nii.gz")
    if max_probs_lookup is not None:
        max_prob = max_probs_lookup[case_id]
    else:
        max_prob = get_max_tumor_probability(predictions_dir / f"{case_id}.npz")

    gt_has_tumor = bool(np.any(gt_mask))
    pred_has_tumor = bool(np.any(pred_mask))

    result = {
        "case_id": case_id,
        "gt_has_tumor": gt_has_tumor,
        "pred_has_tumor": pred_has_tumor,
        "max_tumor_probability": max_prob,
        "dice": None,
        "num_tumor_instances": None,
        "num_detected_instances": None,
    }

    if gt_has_tumor:
        result["dice"] = dice_score(gt_mask, pred_mask)
        num_instances, num_detected = count_tumor_instances_and_detections(gt_mask, pred_mask)
        result["num_tumor_instances"] = num_instances
        result["num_detected_instances"] = num_detected

    return result


def compute_final_metrics(per_case_results: list) -> dict:
    """Combines all 901 individual case results into the five numbers we actually report."""
    positive_cases = [r for r in per_case_results if r["gt_has_tumor"]]
    negative_cases = [r for r in per_case_results if not r["gt_has_tumor"]]

    # P-Sen: of patients who truly have a tumor, what fraction did we find ANY tumor in?
    p_sen = np.mean([r["pred_has_tumor"] for r in positive_cases])

    # Spe: of patients who truly have NO tumor, what fraction did we correctly leave alone?
    spe = np.mean([not r["pred_has_tumor"] for r in negative_cases])

    # T-Sen: of every INDIVIDUAL tumor (not per patient), what fraction did we localize correctly?
    total_instances = sum(r["num_tumor_instances"] for r in positive_cases)
    total_detected = sum(r["num_detected_instances"] for r in positive_cases)
    t_sen = total_detected / total_instances

    # DSC: average overlap score, only over patients who actually have a tumor
    dsc = np.mean([r["dice"] for r in positive_cases])

    # AUC: across ALL 901 cases (positive and negative), does the confidence score separate them well?
    labels = [r["gt_has_tumor"] for r in per_case_results]
    scores = [r["max_tumor_probability"] for r in per_case_results]
    auc = roc_auc_score(labels, scores)

    return {"P-Sen": p_sen, "T-Sen": t_sen, "Spe": spe, "AUC": auc, "DSC": dsc}


if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--per-case-csv", type=Path, default=None,
                         help="where to save the detailed per-case breakdown, for later inspection")
    parser.add_argument("--num-parts", type=int, default=1,
                         help="split the case list into this many independent chunks, to run several "
                              "of these in parallel (like we did for prediction) -- each part only "
                              "evaluates its own slice and saves its own CSV; combine_metrics.py "
                              "merges them afterward")
    parser.add_argument("--part-id", type=int, default=0, help="which chunk this run handles (0-indexed)")
    parser.add_argument("--max-probs-csv", type=Path, default=None,
                         help="output of extract_max_probs.py -- if given, use this small lookup "
                              "instead of loading each case's large .npz file directly")
    args = parser.parse_args()

    if args.per_case_csv is None:
        args.per_case_csv = Path(f"per_case_results_part{args.part_id}.csv") if args.num_parts > 1 \
            else Path("per_case_results.csv")

    max_probs_lookup = None
    if args.max_probs_csv is not None:
        import csv as csv_module
        with open(args.max_probs_csv, newline="") as f:
            max_probs_lookup = {row["case_id"]: float(row["max_tumor_probability"])
                                 for row in csv_module.DictReader(f)}
        print(f"Loaded {len(max_probs_lookup)} pre-extracted max probabilities from {args.max_probs_csv}",
              flush=True)

    case_ids = sorted(p.name.removesuffix(".nii.gz") for p in args.ground_truth_dir.glob("*.nii.gz"))
    if args.num_parts > 1:
        case_ids = case_ids[args.part_id::args.num_parts]  # every Nth case, offset by this part's id
    print(f"Found {len(case_ids)} test cases to evaluate (part {args.part_id}/{args.num_parts}).", flush=True)

    per_case_results = []
    skipped_cases = []
    for i, case_id in enumerate(case_ids, start=1):
        try:
            result = evaluate_case(case_id, args.predictions_dir, args.ground_truth_dir, max_probs_lookup)
            per_case_results.append(result)
        except Exception as e:
            print(f"  SKIPPING {case_id} due to error: {e}", flush=True)
            skipped_cases.append(case_id)
        if i % 20 == 0:
            print(f"[{i}/{len(case_ids)}] evaluated: {case_id}", flush=True)

    if skipped_cases:
        print(f"\nWARNING: skipped {len(skipped_cases)} case(s) due to missing/unreadable files: "
              f"{skipped_cases}", flush=True)

    with open(args.per_case_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_case_results[0].keys()))
        writer.writeheader()
        writer.writerows(per_case_results)
    print(f"Per-case details saved to {args.per_case_csv}", flush=True)

    if args.num_parts > 1:
        print(f"\nThis was part {args.part_id}/{args.num_parts} -- run combine_metrics.py once all "
              f"parts finish to get the final combined metrics.", flush=True)
    else:
        metrics = compute_final_metrics(per_case_results)
        print(f"\nMetrics based on {len(per_case_results)}/{len(case_ids)} cases:", flush=True)
        print("\nFinal metrics on the official PanTS test set:", flush=True)
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}", flush=True)
