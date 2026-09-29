"""CPU-only exporter parity. Set PANTS_TEST_MODEL_FOLDER for the installed-stack test.

This tests restored masks/scores, not GPU forward inference. No model is loaded.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from evaluation.predict_persistent import compact_export, valid_case
from evaluation.predict_and_shrink import shrink_segmentation


class CompactExportTest(unittest.TestCase):
    def test_completed_mask_requires_geometry_and_only_tumor_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.nii.gz"
            output = Path(directory) / "output.nii.gz"
            data = np.zeros((3, 4, 5), np.uint8)
            nib.save(nib.Nifti1Image(data, np.eye(4)), image)
            self.assertFalse(valid_case(output, image, 28))
            data[0, 0, 0] = 28
            nib.save(nib.Nifti1Image(data, np.eye(4)), output)
            self.assertTrue(valid_case(output, image, 28))
            data[1, 1, 1] = 27
            nib.save(nib.Nifti1Image(data, np.eye(4)), output)
            self.assertFalse(valid_case(output, image, 28))
            data[1, 1, 1] = 0
            nib.save(nib.Nifti1Image(data, np.diag([2, 1, 1, 1])), output)
            self.assertFalse(valid_case(output, image, 28))

    @unittest.skipUnless(os.environ.get("PANTS_TEST_MODEL_FOLDER"), "requires actual installed nnU-Net stack")
    def test_exact_mask_and_score_parity_with_official_archive_export(self):
        from nnunetv2.inference.export_prediction import export_prediction_from_logits
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        model = Path(os.environ["PANTS_TEST_MODEL_FOLDER"])
        plans_dict = json.loads((model / "plans.json").read_text())
        dataset = json.loads((model / "dataset.json").read_text())
        # Exercise crop restoration, nontrivial transpose, resampling, organs,
        # rare tumor voxels and close competing logits, not only a trivial argmax.
        plans_dict = copy.deepcopy(plans_dict)
        plans_dict["transpose_forward"] = [2, 0, 1]
        plans_dict["transpose_backward"] = [1, 2, 0]
        plans = PlansManager(plans_dict)
        config = plans.get_configuration("3d_fullres")
        label_manager = plans.get_label_manager(dataset)
        self.assertEqual(label_manager.num_segmentation_heads, 29)
        predictor = SimpleNamespace(plans_manager=plans, configuration_manager=config,
                                    label_manager=label_manager)
        properties = {
            "spacing": [config.spacing[1], config.spacing[2], config.spacing[0]],
            "shape_before_cropping": [8, 9, 10],
            "shape_after_cropping_and_before_resampling": [4, 6, 8],
            "bbox_used_for_cropping": [[1, 5], [2, 8], [1, 9]],
            "sitk_stuff": {"spacing": (1.3, 1.6, 2.1), "origin": (2., 3., 4.),
                           "direction": (1., 0., 0., 0., 1., 0., 0., 0., 1.)},
        }
        rng = np.random.default_rng(20260929)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                logits = rng.normal(0, 1e-4, (29, 3, 5, 7)).astype(np.float32)
                logits[28, 1, 2, 3] = 0.8
                logits[27, 0, 0, 0] = 0.5
                reference = root / f"reference_{index}"
                output = root / f"compact_{index}.nii.gz"
                export_prediction_from_logits(logits.copy(), properties, config, plans, dataset,
                                              str(reference), save_probabilities=True,
                                              num_threads_torch=8)
                reference_mask = reference.with_suffix(".nii.gz")
                shrink_segmentation(reference_mask, 28)
                with np.load(reference.with_suffix(".npz")) as archive:
                    expected_score = float(archive["probabilities"][28].max())
                score = compact_export(logits.copy(), predictor, properties, output, 28)
                expected, actual = nib.load(reference_mask), nib.load(output)
                np.testing.assert_array_equal(np.asanyarray(expected.dataobj),
                                              np.asanyarray(actual.dataobj))
                np.testing.assert_array_equal(expected.affine, actual.affine)
                self.assertEqual(score, expected_score)


if __name__ == "__main__":
    unittest.main()
