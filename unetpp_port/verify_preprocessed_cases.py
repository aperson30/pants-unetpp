"""Verify that every PanTS training case has complete nnU-Net v2 preprocessing output.

This reads filenames and JSON only; it neither rewrites data nor imports PyTorch.
The validated nnU-Net 2.8.1 preprocessor writes .b2nd arrays, not .npz archives.
"""

import argparse
import json
import re
import sys
from pathlib import Path


CASE_RE = re.compile(r"PanTS_\d{8}\Z")


def matching_names(folder: Path, suffix: str, *, nonempty: bool = False) -> set[str]:
    if not folder.is_dir():
        raise ValueError(f"missing directory: {folder}")
    names = set()
    for entry in folder.iterdir():
        if entry.is_file() and entry.name.endswith(suffix):
            if nonempty and entry.stat().st_size == 0:
                raise ValueError(f"empty preprocessed file: {entry}")
            names.add(entry.name[: -len(suffix)])
    return names


def require_same(label: str, actual: set[str], expected: set[str]) -> None:
    if actual != expected:
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise ValueError(
            f"{label}: expected {len(expected)} matching cases, found {len(actual)}; "
            f"missing={missing}, extra={extra}"
        )


def verify(raw_dir: Path, preprocessed_dir: Path, plans_file: Path, expected_count: int) -> None:
    dataset = json.loads((raw_dir / "dataset.json").read_text())
    if dataset["numTraining"] != expected_count:
        raise ValueError(f"dataset.json numTraining={dataset['numTraining']}, expected {expected_count}")
    if dataset["labels"]["pancreatic_lesion"] != 28:
        raise ValueError("pancreatic_lesion must be class 28")

    cases = matching_names(raw_dir / "labelsTr", ".nii.gz")
    if len(cases) != expected_count or any(not CASE_RE.fullmatch(case) for case in cases):
        raise ValueError(f"labelsTr has {len(cases)} cases; expected {expected_count} PanTS IDs")
    require_same(
        "imagesTr",
        matching_names(raw_dir / "imagesTr", ".nii.gz"),
        {f"{case}_0000" for case in cases},
    )

    plans = json.loads(plans_file.read_text())
    data_id = plans["configurations"]["3d_fullres"]["data_identifier"]
    output_dir = preprocessed_dir / data_id
    images = matching_names(output_dir, ".b2nd", nonempty=True)
    segmentations = {case[: -len("_seg")] for case in images if case.endswith("_seg")}
    images = {case for case in images if not case.endswith("_seg")}
    metadata = matching_names(output_dir, ".pkl", nonempty=True)
    require_same("preprocessed image .b2nd", images, cases)
    require_same("preprocessed label _seg.b2nd", segmentations, cases)
    require_same("preprocessed metadata .pkl", metadata, cases)
    print(f"PREPROCESSED_CASES_VERIFIED count={expected_count} data_identifier={data_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("preprocessed_dir", type=Path)
    parser.add_argument("plans_file", type=Path)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    try:
        verify(args.raw_dir, args.preprocessed_dir, args.plans_file, args.expected)
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"PREPROCESSED_CASES_FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
