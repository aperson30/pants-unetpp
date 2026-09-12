"""
Combines the per-part CSV files (from running compute_pants_metrics.py with --num-parts > 1) into
the final five metrics, computed over all cases together.

Run with:
    python combine_metrics.py per_case_results_part0.csv per_case_results_part1.csv ...
"""
import csv
import sys

from compute_pants_metrics import compute_final_metrics

if __name__ == "__main__":
    all_results = []
    for csv_path in sys.argv[1:]:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                row["gt_has_tumor"] = row["gt_has_tumor"] == "True"
                row["pred_has_tumor"] = row["pred_has_tumor"] == "True"
                for key in ["dice", "num_tumor_instances", "num_detected_instances"]:
                    row[key] = None if row[key] == "" else float(row[key])
                row["max_tumor_probability"] = float(row["max_tumor_probability"])
                all_results.append(row)

    print(f"Combined {len(all_results)} total cases from {len(sys.argv) - 1} part file(s).")

    metrics = compute_final_metrics(all_results)
    print("\nFinal metrics on the official PanTS test set:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
