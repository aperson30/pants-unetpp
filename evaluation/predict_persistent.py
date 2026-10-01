"""Candidate single-load test predictor. NOT the production launch default yet.

Preserves nnU-Net's full sliding-window inference, Gaussian blending, mirroring,
fold weights and probability-space export. Avoids restarting the CLI every five
cases and avoids writing/reloading full 29-channel probability archives.
Require real-case parity against predict_and_shrink.py before deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import importlib.metadata
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.predict_and_shrink import (
    read_scores, shrink_segmentation, write_scores, invalidate_pending_scores,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_case(prediction: Path, source: Path, tumor_class: int) -> bool:
    try:
        output, image = nib.load(prediction), nib.load(source)
        labels = np.asanyarray(output.dataobj)
        return (labels.shape == image.shape and labels.ndim == 3
                and np.allclose(output.affine, image.affine, rtol=0, atol=1e-4)
                and bool(np.isin(labels, (0, tumor_class)).all()))
    except (OSError, ValueError, EOFError, nib.filebasedimages.ImageFileError):
        return False


def compact_export(logits, predictor, properties: dict, output: Path,
                   tumor_class: int) -> float:
    # Exactly the helper used by nnUNetv2_predict --save_probabilities. Do not
    # replace softmax-before-argmax with logits argmax or change resampling.
    from nnunetv2.inference.export_prediction import (
        convert_predicted_logits_to_segmentation_with_correct_shape,
    )
    segmentation, probabilities = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, predictor.plans_manager, predictor.configuration_manager,
        predictor.label_manager, properties, return_probabilities=True,
        num_threads_torch=1,
    )
    if probabilities.ndim != 4 or not 0 <= tumor_class < probabilities.shape[0]:
        raise RuntimeError(f"invalid probability shape: {probabilities.shape}")
    tumor_probabilities = probabilities[tumor_class]
    if not np.isfinite(tumor_probabilities).all():
        raise RuntimeError("non-finite tumor probabilities")
    score = float(tumor_probabilities.max())
    if not 0 <= score <= 1:
        raise RuntimeError(f"invalid tumor score: {score}")
    del tumor_probabilities, probabilities
    temporary = output.with_name(output.name + ".partial.nii.gz")
    writer = predictor.plans_manager.image_reader_writer_class()
    writer.write_seg(segmentation, str(temporary), properties)
    shrink_segmentation(temporary, tumor_class)
    os.replace(temporary, output)
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=901)
    parser.add_argument("--tumor-class", type=int, default=28)
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="fp16",
                        help="fp16 matches the existing prediction CLI; bf16 is a protocol choice")
    args = parser.parse_args()
    files = sorted(args.images_dir.glob("*_0000.nii.gz"))
    cases = {path.name[:-12]: path for path in files}
    if len(cases) != args.expected_cases or len(cases) != len(files):
        raise RuntimeError(f"expected {args.expected_cases} unique inputs, got {len(cases)}")
    import torch
    import nnunetv2.inference.predict_from_raw_data as inference_module
    import nnunetv2.inference.export_prediction as export_module
    import nnunetv2.training.nnUNetTrainer as trainer_package
    trainer_directory = Path(trainer_package.__path__[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint_sha256": sha256(args.model_folder / "fold_0/checkpoint_final.pth"),
        "plans_sha256": sha256(args.model_folder / "plans.json"),
        "dataset_sha256": sha256(args.model_folder / "dataset.json"),
        "runner_sha256": sha256(Path(__file__)),
        "shrink_runner_sha256": sha256(Path(__file__).with_name("predict_and_shrink.py")),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nnunetv2": importlib.metadata.version("nnunetv2"),
        "inference_source_sha256": sha256(Path(inference_module.__file__)),
        "export_source_sha256": sha256(Path(export_module.__file__)),
        "custom_trainer_sources": {
            path.name: sha256(path) for path in sorted(trainer_directory.glob("*.py"))
            if path.name.startswith(("nnUNetTrainerBF16", "nnUNetTrainerUNetPlusPlus",
                                     "nnUNetTrainerQualityNeutral", "nnUNetTrainerFullLoss",
                                     "nnUNetTrainerSparse")) or path.name == "unet_plusplus.py"
        },
        "precision": args.precision, "fold": 0, "tile_step_size": 0.5,
        "mirroring": True, "gaussian": True, "tumor_class": args.tumor_class,
        "inputs": {cid: sha256(path) for cid, path in cases.items()},
    }
    manifest_path = args.output_dir / "persistent_prediction_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("prediction provenance changed: do not mix outputs")
    else:
        if any(args.output_dir.glob("*.nii.gz")) or (args.output_dir / "max_tumor_probs.csv").exists():
            raise RuntimeError("existing outputs lack this runner's provenance")
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(manifest_path)
    scores_path = args.output_dir / "max_tumor_probs.csv"
    scores = read_scores(scores_path)
    if scores.keys() - cases.keys():
        raise RuntimeError("unexpected case IDs in score CSV")
    pending = [cid for cid, source in cases.items()
               if cid not in scores or not valid_case(
                   args.output_dir / f"{cid}.nii.gz", source, args.tumor_class)]
    scores = invalidate_pending_scores(scores_path, scores, pending)
    if not pending:
        print(f"all {len(cases)} cases already audited", flush=True)
        return

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    os.environ["nnUNet_compile"] = "false"
    torch.set_num_threads(1)
    torch.set_autocast_dtype("cuda", torch.float16 if args.precision == "fp16" else torch.bfloat16)
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                               perform_everything_on_device=True, device=torch.device("cuda"),
                               allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(args.model_folder), (0,),
                                                   checkpoint_name="checkpoint_final.pth")
    # nnU-Net 2.8.1's own iterator uses a size-one queue per worker. One worker
    # bounds prefetched CT memory; no whole-dataset list of logits/probabilities.
    iterator = predictor._internal_get_data_iterator_from_lists_of_filenames(
        [[str(cases[cid])] for cid in pending], None, None, 1)
    try:
        for index, item in enumerate(iterator):
            cid = pending[index]
            logits = predictor.predict_logits_from_preprocessed_data(item["data"]).cpu().detach().numpy()
            score = compact_export(logits, predictor, item["data_properties"],
                                   args.output_dir / f"{cid}.nii.gz", args.tumor_class)
            del logits, item
            if not valid_case(args.output_dir / f"{cid}.nii.gz", cases[cid], args.tumor_class):
                raise RuntimeError(f"invalid exported mask for {cid}")
            scores[cid] = score
            write_scores(scores_path, scores)
            print(f"completed {index + 1}/{len(pending)}: {cid} score={score}", flush=True)
    finally:
        iterator.close()
        # The upstream iterator does not fully clean up on exception. Run this
        # candidate only as a dedicated Slurm step, so step teardown contains
        # its child processes. Do not embed it in a long-lived Python server.
    missing = [cid for cid in cases if cid not in scores or not valid_case(
        args.output_dir / f"{cid}.nii.gz", cases[cid], args.tumor_class)]
    if missing:
        raise RuntimeError(f"incomplete predictions: {missing[:10]}")
    print(f"verified {len(cases)} complete predictions", flush=True)


if __name__ == "__main__":
    main()
