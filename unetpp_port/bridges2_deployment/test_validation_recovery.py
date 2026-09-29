"""CPU-only tests for backup safety, corruption rejection and input filtering."""
import json
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

import recover_validation as recovery
from snapshot_completed_training import verified_copy


class RecoveryTests(unittest.TestCase):
    def test_backup_refuses_to_overwrite_different_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, destination = root / "source", root / "backup"
            source.write_bytes(b"trained weights")
            first = verified_copy(source, destination)
            self.assertEqual(first, verified_copy(source, destination))
            source.write_bytes(b"different weights")
            with self.assertRaisesRegex(RuntimeError, "refusing overwrite"):
                verified_copy(source, destination)
            self.assertEqual(destination.read_bytes(), b"trained weights")

    def test_prediction_audit_rejects_corruption_geometry_and_bad_labels(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            gt, pred = root / "gt.nii.gz", root / "pred.nii.gz"
            data = np.zeros((8, 9, 10), dtype=np.uint8)
            data[2, 3, 4] = 28
            nib.save(nib.Nifti1Image(data, np.eye(4)), gt)
            nib.save(nib.Nifti1Image(data, np.eye(4)), pred)
            self.assertTrue(recovery.readable_prediction(pred, gt))
            bad_affine = np.eye(4)
            bad_affine[0, 3] = 10
            nib.save(nib.Nifti1Image(data, bad_affine), pred)
            self.assertFalse(recovery.readable_prediction(pred, gt))
            for value in (29, 0.5, np.nan):
                bad = data.astype(np.float32)
                bad[2, 3, 4] = value
                nib.save(nib.Nifti1Image(bad, np.eye(4)), pred)
                self.assertFalse(recovery.readable_prediction(pred, gt))
            pred.write_bytes(b"not a nifti")
            self.assertFalse(recovery.readable_prediction(pred, gt))
            nib.save(nib.Nifti1Image(data, np.eye(4)), pred)
            content = pred.read_bytes()
            pred.write_bytes(content[:len(content)//2])
            self.assertFalse(recovery.readable_prediction(pred, gt))

    def test_resume_filters_inputs_but_scores_the_full_fold(self):
        self._exercise_resume()

    def test_checkpoint_drift_aborts_before_loading_model(self):
        self._exercise_resume(stale=True)

    def _exercise_resume(self, stale=False):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            fold, pre = root / "model" / "fold_0", root / "pre"
            fold.mkdir(parents=True)
            pre.mkdir()
            (fold / "validation").mkdir()
            checkpoint = fold / "checkpoint_final.pth"
            checkpoint.write_bytes(b"final epoch 1000")
            plans = {"configurations": {"3d_fullres": {"batch_size": 4}}}
            dataset = {"labels": {"pancreatic_lesion": 28}}
            (fold.parent / "plans.json").write_text(json.dumps(plans))
            (fold.parent / "dataset.json").write_text(json.dumps(dataset))
            val = [f"PanTS_{index:08d}" for index in range(1800)]
            train = [f"Train_{index}" for index in range(7200)]
            loaded = []
            seen = []

            class Dataset:
                def __init__(self, folder, identifiers, **kwargs):
                    self.identifiers = identifiers

            def load_checkpoint(path):
                loaded.append(path)

            def validate(save_probabilities):
                self.assertFalse(save_probabilities)
                self.assertEqual(trainer.do_split(), (train, val))
                selected = trainer.dataset_class(str(pre), val, folder_with_segs_from_previous_stage=None)
                seen.extend(selected.identifiers)
                summary = {"metric_per_case": [{"prediction_file": f"{key}.nii.gz"} for key in val],
                           "mean": {"28": {"Dice": 0.1}}}
                (fold / "validation" / "summary.json").write_text(json.dumps(summary))

            trainer = SimpleNamespace(output_folder=str(fold), plans_manager=SimpleNamespace(plans=plans),
                                      dataset_json=dataset, current_epoch=1000,
                                      configuration_manager=SimpleNamespace(next_stage_names=None),
                                      preprocessed_dataset_folder_base=str(pre), dataset_class=Dataset,
                                      load_checkpoint=load_checkpoint, do_split=lambda: (train, val),
                                      perform_actual_validation=validate)
            manifest = {"cells": {"plain_ds_off": {"trainer": recovery.TRAINER,
                         "training_complete_from_logs": True,
                         "checkpoint": {"source": str(checkpoint), "sha256": recovery.sha256(checkpoint)}}}}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            if stale:
                checkpoint.write_bytes(b"changed after manifest")
            # Isolate orchestration tests from PyTorch import/compiler startup.
            # The production API signatures are separately checked against the installed source.
            fake_torch = SimpleNamespace(no_grad=nullcontext, backends=SimpleNamespace(cudnn=SimpleNamespace()))
            fake_loader = SimpleNamespace(get_trainer_from_args=lambda *args, **kwargs: trainer)
            with patch.dict(sys.modules, {"torch": fake_torch,
                                          "nnunetv2": SimpleNamespace(),
                                          "nnunetv2.run": SimpleNamespace(),
                                          "nnunetv2.run.run_training": fake_loader}), patch.object(
                    recovery, "readable_prediction", side_effect=lambda path, gt: path.stem.startswith("PanTS_00000000")):
                if stale:
                    with self.assertRaisesRegex(AssertionError, "checkpoint changed"):
                        recovery.run(manifest_path)
                    self.assertEqual(loaded, [])
                else:
                    recovery.run(manifest_path)
                    self.assertEqual(seen, val[1:])
                    self.assertEqual(loaded, [str(checkpoint)])


if __name__ == "__main__":
    unittest.main()
