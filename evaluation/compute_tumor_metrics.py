"""Compute the project's five tumor metrics with fail-closed input auditing.

The public PanTS materials do not fully specify connectivity, DSC averaging, or
the continuous AUC score. The output therefore records every such choice.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import generate_binary_structure, label as cc_label
from sklearn.metrics import roc_auc_score

TUMOR_CLASS_DEFAULT = 28


def case_id(path: Path) -> str:
    if not path.name.endswith(".nii.gz"):
        raise ValueError(f"not a .nii.gz file: {path}")
    return path.name[:-7]


def load_label(path: Path) -> tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if not np.isfinite(data).all():
        raise ValueError(f"non-finite label values in {path}")
    return data, image


def dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    denominator = int(pred_mask.sum()) + int(gt_mask.sum())
    if denominator == 0:
        return float("nan")
    return 2.0 * int(np.logical_and(pred_mask, gt_mask).sum()) / denominator


def tumor_wise_detection(
    pred_mask: np.ndarray, gt_mask: np.ndarray, connectivity: int
) -> tuple[int, int]:
    structure = generate_binary_structure(rank=3, connectivity=connectivity)
    components, n_components = cc_label(gt_mask, structure=structure)
    detected = sum(
        bool(np.logical_and(components == component_id, pred_mask).any())
        for component_id in range(1, n_components + 1)
    )
    return int(n_components), int(detected)


def read_probabilities(path: Path) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"case_id", "max_tumor_probability"}
        if not reader.fieldnames or not expected.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(expected)}")
        for row_number, row in enumerate(reader, start=2):
            cid = row["case_id"]
            if cid in probabilities:
                raise ValueError(f"duplicate probability for {cid} at {path}:{row_number}")
            value = float(row["max_tumor_probability"])
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid probability for {cid}: {value}")
            probabilities[cid] = value
    return probabilities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--tumor-class", type=int, default=TUMOR_CLASS_DEFAULT)
    parser.add_argument("--probs-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--per-case-csv", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=901)
    parser.add_argument(
        "--connectivity", type=int, choices=(1, 2, 3), default=1,
        help="1=6-neighbor, 2=18-neighbor, 3=26-neighbor; default preserves prior scoring",
    )
    parser.add_argument("--affine-atol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_paths = sorted(args.pred_dir.glob("*.nii.gz"))
    gt_paths = sorted(args.labels_dir.glob("*.nii.gz"))
    predictions = {case_id(path): path for path in pred_paths}
    ground_truth = {case_id(path): path for path in gt_paths}

    if len(predictions) != len(pred_paths) or len(ground_truth) != len(gt_paths):
        raise RuntimeError("duplicate case IDs after stripping .nii.gz")
    if len(predictions) != args.expected_cases:
        raise RuntimeError(f"expected {args.expected_cases} predictions, found {len(predictions)}")
    missing_gt = sorted(predictions.keys() - ground_truth.keys())
    missing_predictions = sorted(ground_truth.keys() - predictions.keys())
    if missing_gt or missing_predictions:
        raise RuntimeError(
            f"prediction/GT ID mismatch: missing_gt={missing_gt[:10]}, "
            f"missing_predictions={missing_predictions[:10]}"
        )

    probabilities = read_probabilities(args.probs_csv)
    missing_probs = sorted(predictions.keys() - probabilities.keys())
    extra_probs = sorted(probabilities.keys() - predictions.keys())
    if missing_probs or extra_probs:
        raise RuntimeError(
            f"probability/prediction ID mismatch: missing={missing_probs[:10]}, extra={extra_probs[:10]}"
        )

    rows: list[dict[str, object]] = []
    positive_dice: list[float] = []
    p_hits = p_total = spe_hits = spe_total = 0
    detected_tumors = true_tumors = 0

    for index, cid in enumerate(sorted(predictions), start=1):
        pred, pred_image = load_label(predictions[cid])
        gt, gt_image = load_label(ground_truth[cid])
        if pred.shape != gt.shape:
            raise RuntimeError(f"shape mismatch for {cid}: prediction {pred.shape}, GT {gt.shape}")
        if not np.allclose(pred_image.affine, gt_image.affine, rtol=0, atol=args.affine_atol):
            raise RuntimeError(f"affine mismatch for {cid}")

        pred_mask = pred == args.tumor_class
        gt_mask = gt == args.tumor_class
        gt_positive = bool(gt_mask.any())
        pred_positive = bool(pred_mask.any())
        case_dice = dice(pred_mask, gt_mask)
        n_true, n_detected = (
            tumor_wise_detection(pred_mask, gt_mask, args.connectivity)
            if gt_positive else (0, 0)
        )

        if gt_positive:
            p_total += 1
            p_hits += pred_positive
            positive_dice.append(case_dice)
            true_tumors += n_true
            detected_tumors += n_detected
        else:
            spe_total += 1
            spe_hits += not pred_positive

        rows.append({
            "case_id": cid,
            "gt_positive": int(gt_positive),
            "pred_positive": int(pred_positive),
            "gt_tumor_voxels": int(gt_mask.sum()),
            "pred_tumor_voxels": int(pred_mask.sum()),
            "true_tumors": n_true,
            "detected_tumors": n_detected,
            "dice": "" if np.isnan(case_dice) else case_dice,
            "max_tumor_probability": probabilities[cid],
        })
        if index % 200 == 0:
            print(f"processed {index}/{len(predictions)}", flush=True)

    y_true = [int(row["gt_positive"]) for row in rows]
    y_score = [float(row["max_tumor_probability"]) for row in rows]
    if len(set(y_true)) != 2:
        raise RuntimeError("AUC requires both tumor-positive and tumor-negative cases")

    results = {
        "protocol": {
            "tumor_class": args.tumor_class,
            "prediction_mask": "nnU-Net argmax label equals tumor_class",
            "DSC": "tumor-only mean over GT-positive cases",
            "P_Sen": "GT-positive patient detected if any tumor voxel is predicted anywhere",
            "T_Sen": "GT component detected by at least one overlapping predicted voxel",
            "T_Sen_connectivity": {1: "6-neighbor", 2: "18-neighbor", 3: "26-neighbor"}[args.connectivity],
            "Spe": "GT-negative patient is negative only if zero tumor voxels are predicted",
            "AUC_score": "maximum softmax tumor-class probability over the full volume",
            "note": "Project protocol; public PanTS materials do not fully specify all choices.",
        },
        "n_cases_evaluated": len(rows),
        "DSC_tumor_mean": float(np.mean(positive_dice)),
        "DSC_tumor_n_positive_cases": len(positive_dice),
        "P_Sen": p_hits / p_total,
        "P_Sen_n_positive_cases": p_total,
        "T_Sen": detected_tumors / true_tumors,
        "T_Sen_n_true_tumors": true_tumors,
        "Spe": spe_hits / spe_total,
        "Spe_n_negative_cases": spe_total,
        "AUC": float(roc_auc_score(y_true, y_score)),
        "AUC_n_cases": len(y_true),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.per_case_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.per_case_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with args.out_json.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
