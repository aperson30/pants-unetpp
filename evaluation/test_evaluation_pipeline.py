from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.predict_and_shrink import shrink_segmentation


class EvaluationPipelineTest(unittest.TestCase):
    def test_metrics_and_tumor_only_shrink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            pred_dir, gt_dir = root / "pred", root / "gt"
            pred_dir.mkdir()
            gt_dir.mkdir()
            affine = np.eye(4)

            positive_gt = np.zeros((4, 4, 4), dtype=np.uint8)
            positive_gt[0, 0, 0] = 28
            positive_gt[2, 2, 2] = 28
            positive_pred = np.zeros_like(positive_gt)
            positive_pred[0, 0, 0] = 28
            positive_pred[3, 3, 3] = 28
            negative = np.zeros_like(positive_gt)

            for cid, pred, gt in (
                ("PanTS_00009001", positive_pred, positive_gt),
                ("PanTS_00009002", negative, negative),
            ):
                nib.save(nib.Nifti1Image(pred, affine), pred_dir / f"{cid}.nii.gz")
                nib.save(nib.Nifti1Image(gt, affine), gt_dir / f"{cid}.nii.gz")

            # Ensure tumor-only compaction deletes an organ label but preserves tumor.
            compact_path = pred_dir / "PanTS_00009001.nii.gz"
            compact_image = nib.load(compact_path)
            compact_data = np.asanyarray(compact_image.dataobj).copy()
            compact_data[1, 1, 1] = 17
            nib.save(nib.Nifti1Image(compact_data, affine), compact_path)
            shrink_segmentation(compact_path, 28)
            self.assertEqual(set(np.unique(np.asanyarray(nib.load(compact_path).dataobj))), {0, 28})

            probs = root / "probs.csv"
            with probs.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["case_id", "max_tumor_probability"])
                writer.writerow(["PanTS_00009001", 0.9])
                writer.writerow(["PanTS_00009002", 0.1])

            output, per_case = root / "metrics.json", root / "per_case.csv"
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("compute_tumor_metrics.py")),
                "--pred-dir", str(pred_dir), "--labels-dir", str(gt_dir),
                "--probs-csv", str(probs), "--out-json", str(output),
                "--per-case-csv", str(per_case), "--expected-cases", "2",
                "--connectivity", "1",
            ], check=True)
            result = json.loads(output.read_text())
            self.assertEqual(result["n_cases_evaluated"], 2)
            self.assertAlmostEqual(result["DSC_tumor_mean"], 0.5)
            self.assertAlmostEqual(result["P_Sen"], 1.0)
            self.assertAlmostEqual(result["T_Sen"], 0.5)
            self.assertAlmostEqual(result["Spe"], 1.0)
            self.assertAlmostEqual(result["AUC"], 1.0)


if __name__ == "__main__":
    unittest.main()
