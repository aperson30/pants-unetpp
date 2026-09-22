"""Run bounded-disk nnU-Net inference and retain an auditable tumor score.

The script is restartable but fail-closed: a failed batch aborts, duplicate CSV
rows are rejected, and success requires one valid segmentation and probability
for every input case. Optionally, completed segmentations are rewritten to
contain only labels 0 and ``tumor_class`` to make persistent storage tiny while
preserving all five tumor metrics.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


def read_scores(path: Path) -> dict[str, float]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    scores: dict[str, float] = {}
    with path.open(newline="") as handle:
        for line, row in enumerate(csv.DictReader(handle), start=2):
            cid = row["case_id"]
            if cid in scores:
                raise RuntimeError(f"duplicate {cid} in {path}:{line}")
            value = float(row["max_tumor_probability"])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise RuntimeError(f"invalid probability for {cid}: {value}")
            scores[cid] = value
    return scores


def write_scores(path: Path, scores: dict[str, float]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "max_tumor_probability"])
        writer.writerows((cid, scores[cid]) for cid in sorted(scores))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def valid_segmentation(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = np.asanyarray(nib.load(str(path)).dataobj)
        return data.ndim == 3 and bool(np.isfinite(data).all())
    except Exception:
        return False


def shrink_segmentation(path: Path, tumor_class: int) -> None:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    tumor_only = np.where(data == tumor_class, tumor_class, 0).astype(np.uint8)
    temporary = path.with_name(path.name + ".tmp.nii.gz")
    nib.save(nib.Nifti1Image(tumor_only, image.affine, image.header), str(temporary))
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("-tr", required=True)
    parser.add_argument("-p", required=True)
    parser.add_argument("-f", required=True)
    parser.add_argument("--max-probs-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-parts", type=int, default=1)
    parser.add_argument("--part-id", type=int, default=0)
    parser.add_argument("--tumor-class", type=int, default=28)
    parser.add_argument("--expected-cases", type=int, default=901)
    parser.add_argument("--shrink-segmentations-to-tumor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_parts < 1 or not 0 <= args.part_id < args.num_parts:
        raise ValueError("invalid batch/partition arguments")

    all_cases = sorted(
        path.name[:-12] for path in args.images_dir.glob("*_0000.nii.gz")
    )
    if len(all_cases) != args.expected_cases:
        raise RuntimeError(f"expected {args.expected_cases} input cases, found {len(all_cases)}")
    case_ids = all_cases[args.part_id::args.num_parts]
    expected_part = len(range(args.part_id, len(all_cases), args.num_parts))
    if len(case_ids) != expected_part:
        raise AssertionError("partitioning error")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.max_probs_csv.parent.mkdir(parents=True, exist_ok=True)
    scores = read_scores(args.max_probs_csv)
    extra_scores = scores.keys() - set(all_cases)
    if extra_scores:
        raise RuntimeError(f"score CSV contains unexpected cases: {sorted(extra_scores)[:10]}")

    pending: list[str] = []
    for cid in case_ids:
        segmentation = args.output_dir / f"{cid}.nii.gz"
        if cid in scores and valid_segmentation(segmentation):
            continue
        # A partial segmentation without its score makes --continue_prediction skip
        # the case, so remove all partial artifacts and recompute it atomically.
        for suffix in (".nii.gz", ".npz", ".pkl"):
            path = args.output_dir / f"{cid}{suffix}"
            if path.exists():
                path.unlink()
        pending.append(cid)

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        batch_number = start // args.batch_size + 1
        print(f"part {args.part_id}: batch {batch_number}, {len(batch)} cases", flush=True)
        with tempfile.TemporaryDirectory(
            prefix=f"pants_predict_{args.part_id}_", dir=os.environ.get("TMPDIR")
        ) as temp_name:
            temp_input = Path(temp_name)
            for cid in batch:
                source = args.images_dir / f"{cid}_0000.nii.gz"
                if not source.is_file():
                    raise FileNotFoundError(source)
                (temp_input / source.name).symlink_to(source.resolve())

            command = [
                "nnUNetv2_predict", "-i", str(temp_input), "-o", str(args.output_dir),
                "-d", args.dataset, "-c", args.config, "-tr", args.tr,
                "-p", args.p, "-f", args.f, "--save_probabilities",
                "--continue_prediction",
            ]
            subprocess.run(command, check=True)

        for cid in batch:
            segmentation = args.output_dir / f"{cid}.nii.gz"
            probability_path = args.output_dir / f"{cid}.npz"
            if not valid_segmentation(segmentation):
                raise RuntimeError(f"missing or corrupt segmentation for {cid}")
            if not probability_path.is_file():
                raise RuntimeError(f"missing probability archive for {cid}")
            with np.load(probability_path, allow_pickle=False) as archive:
                probabilities = archive["probabilities"]
                if probabilities.ndim != 4 or args.tumor_class >= probabilities.shape[0]:
                    raise RuntimeError(f"unexpected probability shape for {cid}: {probabilities.shape}")
                tumor = probabilities[args.tumor_class]
                if not np.isfinite(tumor).all():
                    raise RuntimeError(f"non-finite tumor probabilities for {cid}")
                score = float(tumor.max())
            if not 0 <= score <= 1:
                raise RuntimeError(f"out-of-range tumor probability for {cid}: {score}")
            if args.shrink_segmentations_to_tumor:
                shrink_segmentation(segmentation, args.tumor_class)
            scores[cid] = score
            write_scores(args.max_probs_csv, scores)
            probability_path.unlink()

    missing = [
        cid for cid in case_ids
        if cid not in scores or not valid_segmentation(args.output_dir / f"{cid}.nii.gz")
    ]
    if missing:
        raise RuntimeError(f"inference incomplete for {len(missing)} cases: {missing[:10]}")
    print(f"part {args.part_id}: verified {len(case_ids)}/{len(case_ids)} complete", flush=True)


if __name__ == "__main__":
    main()
