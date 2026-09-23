"""Small synthetic negative tests for the real-data staging guard."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("verify_preprocessed_cases.py")


class PreprocessedCaseVerificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.raw = root / "raw"
        self.preprocessed = root / "preprocessed"
        self.plans = self.preprocessed / "nnUNetPlansBS4.json"
        (self.raw / "imagesTr").mkdir(parents=True)
        (self.raw / "labelsTr").mkdir()
        (self.preprocessed / "nnUNetPlans_3d_fullres").mkdir(parents=True)
        (self.raw / "dataset.json").write_text(
            json.dumps({"numTraining": 3, "labels": {"pancreatic_lesion": 28}})
        )
        self.plans.write_text(
            json.dumps({"configurations": {"3d_fullres": {
                "data_identifier": "nnUNetPlans_3d_fullres"}}})
        )
        for index in range(1, 4):
            case = f"PanTS_{index:08d}"
            (self.raw / "imagesTr" / f"{case}_0000.nii.gz").touch()
            (self.raw / "labelsTr" / f"{case}.nii.gz").touch()
            for suffix in (".b2nd", "_seg.b2nd", ".pkl"):
                (self.preprocessed / "nnUNetPlans_3d_fullres" / f"{case}{suffix}").write_bytes(b"data")

    def run_guard(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.raw), str(self.preprocessed),
             str(self.plans), "--expected", "3"],
            capture_output=True, text=True, check=False,
        )

    def test_complete_cases_pass(self) -> None:
        result = self.run_guard()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PREPROCESSED_CASES_VERIFIED", result.stdout)

    def test_missing_image_fails(self) -> None:
        (self.preprocessed / "nnUNetPlans_3d_fullres" / "PanTS_00000001.b2nd").unlink()
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preprocessed image", result.stderr)

    def test_missing_segmentation_fails(self) -> None:
        (self.preprocessed / "nnUNetPlans_3d_fullres" / "PanTS_00000001_seg.b2nd").unlink()
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preprocessed label", result.stderr)

    def test_missing_metadata_fails(self) -> None:
        (self.preprocessed / "nnUNetPlans_3d_fullres" / "PanTS_00000001.pkl").unlink()
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("preprocessed metadata", result.stderr)

    def test_wrong_case_id_fails_even_with_correct_count(self) -> None:
        (self.preprocessed / "nnUNetPlans_3d_fullres" / "PanTS_00000001.b2nd").rename(
            self.preprocessed / "nnUNetPlans_3d_fullres" / "PanTS_00000004.b2nd"
        )
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing=", result.stderr)

    def test_empty_preprocessed_file_fails(self) -> None:
        (self.preprocessed / "nnUNetPlans_3d_fullres" / "PanTS_00000001.b2nd").write_bytes(b"")
        result = self.run_guard()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("empty preprocessed file", result.stderr)


if __name__ == "__main__":
    unittest.main()
