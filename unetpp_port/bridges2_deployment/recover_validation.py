"""Resume the interrupted plain U-Net validation with unchanged nnU-Net inference.

Only readable predictions tied to the backed-up final checkpoint are reused.
nnU-Net's installed validator still performs inference, export and scoring;
this script filters its input cases, rather than implementing another predictor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from snapshot_completed_training import sha256


TRAINER = "nnUNetTrainerBF16NoDeepSupervisionSparseValidation"


def readable_prediction(prediction: Path, ground_truth: Path) -> bool:
    import nibabel as nib
    import numpy as np
    from nibabel.filebasedimages import ImageFileError

    try:
        pred, gt = nib.load(str(prediction)), nib.load(str(ground_truth))
        if len(pred.shape) != 3 or pred.shape != gt.shape:
            return False
        if not np.allclose(pred.affine, gt.affine, rtol=1e-5, atol=1e-3):
            return False
        # Force full decompression; a valid header does not detect a truncated gzip.
        data = np.asanyarray(pred.dataobj)
        return bool(np.isfinite(data).all() and (data >= 0).all() and
                    (data <= 28).all() and (data == np.floor(data)).all())
    except (OSError, ValueError, EOFError, ImageFileError):
        return False


def run(manifest_path: Path) -> None:
    import torch
    from nnunetv2.run.run_training import get_trainer_from_args

    manifest = json.loads(manifest_path.read_text())
    record = manifest["cells"]["plain_ds_off"]
    assert record["trainer"] == TRAINER
    assert record["training_complete_from_logs"]
    trainer = get_trainer_from_args("1", "3d_fullres", 0, TRAINER, "nnUNetPlansBS4")
    fold = Path(trainer.output_folder)
    checkpoint = fold / "checkpoint_final.pth"
    assert checkpoint.resolve() == Path(record["checkpoint"]["source"]).resolve()
    expected_hash = record["checkpoint"]["sha256"]
    assert sha256(checkpoint) == expected_hash, "final checkpoint changed"
    saved_plans = json.loads((fold.parent / "plans.json").read_text())
    current_plans = dict(trainer.plans_manager.plans)
    # This loader-only field does not change preprocessing or inference.
    saved_plans.pop("continue_training", None)
    current_plans.pop("continue_training", None)
    assert current_plans == saved_plans, "restaged plans differ from training"
    assert trainer.dataset_json == json.loads((fold.parent / "dataset.json").read_text())
    trainer.load_checkpoint(str(checkpoint))
    assert trainer.current_epoch == 1000, trainer.current_epoch
    assert trainer.configuration_manager.next_stage_names is None
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    train_keys, val_keys = trainer.do_split()
    assert len(train_keys) == 7200 and len(val_keys) == 1800
    validation = fold / "validation"
    validation.mkdir(exist_ok=True)
    actual = {path.name.removesuffix(".nii.gz") for path in validation.glob("*.nii.gz")}
    assert actual <= set(val_keys), "unexpected validation outputs"
    gt_folder = Path(trainer.preprocessed_dataset_folder_base) / "gt_segmentations"
    pending = []
    for index, key in enumerate(val_keys, start=1):
        if not readable_prediction(validation / f"{key}.nii.gz", gt_folder / f"{key}.nii.gz"):
            pending.append(key)
        if index % 100 == 0:
            print(f"audited {index}/1800 existing predictions", flush=True)
    original_dataset_class = trainer.dataset_class
    # Cache the original full fold before filtering the inference dataset.
    # Scoring still sees all 1,800 predictions, including reused cases.
    trainer.do_split = lambda: (train_keys, val_keys)

    def remaining_dataset(folder, identifiers, **kwargs):
        assert set(identifiers) == set(val_keys)
        return original_dataset_class(folder, pending, **kwargs)

    trainer.dataset_class = remaining_dataset
    print(f"VALIDATION_RESUME reused={1800-len(pending)} pending={len(pending)} "
          f"checkpoint_sha256={expected_hash}", flush=True)
    with torch.no_grad():
        trainer.perform_actual_validation(save_probabilities=False)
    assert sha256(checkpoint) == expected_hash, "checkpoint changed during validation"
    summary = json.loads((validation / "summary.json").read_text())
    scored = {Path(item["prediction_file"]).name.removesuffix(".nii.gz")
              for item in summary["metric_per_case"]}
    assert len(summary["metric_per_case"]) == 1800 and scored == set(val_keys)
    assert "28" in summary["mean"]
    print("VALIDATION_RECOVERY_COMPLETE all 1800 cases scored; checkpoint unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint_manifest)
