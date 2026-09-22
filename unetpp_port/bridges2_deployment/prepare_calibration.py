"""Build a small, real PanTS calibration set without downloading either full archive.

The PanTS image and label tarballs use independent randomized case orderings. We stream a bounded
prefix of both archives, retain their intersection, and deliberately include both tumor-positive
and tumor-negative cases according to the public metadata. This is calibration data only: it is
used to exercise the exact trainer/data path, not to estimate model quality.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

import pandas as pd


CASE_RE = re.compile(r"PanTS_\d{8}")
IMAGE_URL = (
    "https://huggingface.co/datasets/BodyMaps/PanTSMini/resolve/main/"
    "PanTSMini_ImageTr_00000001_00001000.tar.gz?download=true"
)
LABEL_URL = "https://www.cs.jhu.edu/~zongwei/dataset/PanTSMini_Label.tar.gz"


def _stream_cases(url: str, destination: Path, limit: int, kind: str) -> list[str]:
    """Stream complete cases from the front of a tar.gz, normalizing away archive prefixes."""
    destination.mkdir(parents=True, exist_ok=True)
    seen: list[str] = []
    seen_set: set[str] = set()
    request = urllib.request.Request(url, headers={"User-Agent": "pants-unetpp-calibration/1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                match = CASE_RE.search(member.name)
                if match is None:
                    continue
                case_id = match.group(0)
                if case_id not in seen_set:
                    if len(seen) >= limit:
                        break
                    seen.append(case_id)
                    seen_set.add(case_id)

                suffix = PurePosixPath(member.name).parts
                case_index = suffix.index(case_id)
                relative = Path(*suffix[case_index + 1 :])
                if not relative.parts:
                    (destination / case_id).mkdir(parents=True, exist_ok=True)
                    continue
                output = destination / case_id / relative
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Unable to read {member.name}")
                    with source, output.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                elif member.issym() or member.islnk():
                    raise RuntimeError(f"Refusing link member in trusted archive: {member.name}")
    print(f"streamed {len(seen)} {kind} cases", flush=True)
    return seen


def _validate_case(case_dir: Path, kind: str) -> None:
    required = case_dir / ("ct.nii.gz" if kind == "image" else "segmentations/pancreas.nii.gz")
    if not required.is_file() or required.stat().st_size == 0:
        raise RuntimeError(f"Incomplete {kind} case {case_dir.name}: missing {required}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prefix-cases", type=int, default=200)
    parser.add_argument("--selected-cases", type=int, default=9)
    parser.add_argument("--tumor-cases", type=int, default=4)
    args = parser.parse_args()

    root = args.root.resolve()
    image_dir = root / "ImageTr"
    label_dir = root / "LabelTr"
    if root.exists():
        # The caller gives this script a calibration-only directory. Refuse to touch anything that
        # does not carry our marker, so a typo can never remove a real dataset.
        marker = root / ".pants_calibration_staging"
        if not marker.is_file():
            raise RuntimeError(f"Refusing existing unmarked directory: {root}")
        shutil.rmtree(image_dir, ignore_errors=True)
        shutil.rmtree(label_dir, ignore_errors=True)
    else:
        root.mkdir(parents=True)
        (root / ".pants_calibration_staging").write_text("calibration only\n", encoding="utf-8")

    image_cases = _stream_cases(IMAGE_URL, image_dir, args.prefix_cases, "image")
    label_cases = _stream_cases(LABEL_URL, label_dir, args.prefix_cases, "label")
    overlap = sorted(set(image_cases) & set(label_cases))
    if len(overlap) < args.selected_cases:
        raise RuntimeError(
            f"Only {len(overlap)} matched cases; increase --prefix-cases (currently {args.prefix_cases})"
        )
    for case_id in overlap:
        _validate_case(image_dir / case_id, "image")
        _validate_case(label_dir / case_id, "label")

    metadata = pd.read_excel(args.metadata, sheet_name="PanTS_metadata")
    tumor_by_case = {
        str(row["PanTS ID"]): bool(int(row["tumor?"]))
        for _, row in metadata.iterrows()
        if pd.notna(row["tumor?"])
    }
    positives = [case for case in overlap if tumor_by_case.get(case, False)]
    negatives = [case for case in overlap if not tumor_by_case.get(case, False)]
    n_positive = min(args.tumor_cases, len(positives))
    selected = positives[:n_positive] + negatives[: args.selected_cases - n_positive]
    if len(selected) < args.selected_cases:
        remaining = [case for case in overlap if case not in selected]
        selected.extend(remaining[: args.selected_cases - len(selected)])
    if not any(tumor_by_case.get(case, False) for case in selected):
        raise RuntimeError("Matched calibration subset contains no tumor-positive case")

    selected_set = set(selected)
    for folder in (image_dir, label_dir):
        for case_dir in folder.iterdir():
            if case_dir.is_dir() and case_dir.name not in selected_set:
                shutil.rmtree(case_dir)
    (root / "selected_cases.txt").write_text(
        "\n".join(f"{case}\ttumor={int(tumor_by_case.get(case, False))}" for case in selected) + "\n",
        encoding="utf-8",
    )
    # The converter expects both split roots even though this calibration has no test cases.
    (root / "ImageTe").mkdir(exist_ok=True)
    (root / "LabelTe").mkdir(exist_ok=True)
    print(f"selected {len(selected)} matched cases ({sum(tumor_by_case.get(c, False) for c in selected)} tumor)")
    print("CALIBRATION_STREAM_DONE", flush=True)


if __name__ == "__main__":
    main()
