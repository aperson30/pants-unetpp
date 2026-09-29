"""Back up final weights and collect compact, auditable progress evidence.

Run after trainers have exited. This never changes a source checkpoint or a
prediction. Large weights remain outside Git; the report contains their hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil


TRAINERS = {
    "unetpp_ds_on": "nnUNetTrainerUNetPlusPlusSparseValidation",
    "unetpp_ds_off": "nnUNetTrainerUNetPlusPlusNoDeepSupervisionSparseValidation",
    "plain_ds_on": "nnUNetTrainerBF16SparseValidation",
    "plain_ds_off": "nnUNetTrainerBF16NoDeepSupervisionSparseValidation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    return value


def verified_copy(source: Path, destination: Path) -> dict:
    before = source.stat()
    digest = sha256(source)
    if destination.exists():
        if sha256(destination) != digest:
            raise RuntimeError(f"existing backup differs; refusing overwrite: {destination}")
    else:
        temporary = destination.with_name(destination.name + ".incomplete")
        if temporary.exists():
            raise RuntimeError(f"review prior incomplete backup first: {temporary}")
        shutil.copyfile(source, temporary)
        if sha256(temporary) != digest:
            raise RuntimeError(f"backup hash mismatch: {temporary}")
        temporary.replace(destination)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed during backup: {source}")
    return {"source": str(source), "backup": str(destination),
            "bytes": after.st_size, "sha256": digest, "backup_verified": True}


def collect(root: Path, backup: Path, report: Path) -> dict:
    backup.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    result = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
              "root": str(root), "cells": {},
              "note": "Validation summaries are not the 901-case test results or the five project tumor metrics."}
    for tag, trainer in TRAINERS.items():
        model = root / "results" / "Dataset001_PanTS" / f"{trainer}__nnUNetPlansBS4__3d_fullres"
        fold = model / "fold_0"
        checkpoint = fold / "checkpoint_final.pth"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError(f"missing final checkpoint: {checkpoint}")
        cell_backup = backup / tag
        cell_backup.mkdir(exist_ok=True)
        cell = {"trainer": trainer, "checkpoint": verified_copy(checkpoint, cell_backup / checkpoint.name)}
        logs = sorted(fold.glob("training_log*.txt"))
        evidence = []
        for path in logs:
            for line in path.read_text(errors="replace").splitlines():
                if any(marker in line for marker in ("Epoch 999", "Training done.", "Validation complete")):
                    evidence.append(line)
        cell["completion_log_evidence"] = evidence
        cell["training_complete_from_logs"] = any("Epoch 999" in line for line in evidence) and any(
            "Training done." in line for line in evidence)
        if not cell["training_complete_from_logs"]:
            raise RuntimeError(f"missing 1000-epoch completion evidence: {tag}")
        for name in ("plans.json", "dataset.json", "dataset_fingerprint.json"):
            source = model / name
            if source.exists():
                verified_copy(source, cell_backup / name)
        source = fold / "debug.json"
        if source.exists():
            verified_copy(source, cell_backup / source.name)
        validation = fold / "validation"
        predictions = sorted(validation.glob("PanTS_*.nii.gz"))
        cell["validation_prediction_count"] = len(predictions)
        cell["validation_prediction_count_is_not_a_readability_audit"] = True
        summary_path = validation / "summary.json"
        cell["validation_summary_present"] = summary_path.is_file()
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            cell["validation_summary_sha256"] = sha256(summary_path)
            cell["validation_summary_case_count"] = len(summary["metric_per_case"])
            cell["class_28_validation_mean"] = finite_json(summary["mean"]["28"])
            tumor_cases = []
            for entry in summary["metric_per_case"]:
                tumor_cases.append({"case_id": Path(entry["prediction_file"]).name.removesuffix(".nii.gz"),
                                    "metrics": finite_json(entry["metrics"]["28"])})
            (report / f"{tag}_class28_validation.json").write_text(
                json.dumps(tumor_cases, indent=2, allow_nan=False) + "\n")
        result["cells"][tag] = cell
        print(f"{tag}: final checkpoint backed up; {len(predictions)}/1800 predictions; "
              f"summary={cell['validation_summary_present']}", flush=True)
    output = report / "checkpoint_manifest.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    shutil.copyfile(output, backup / output.name)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backup-tag", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not args.backup_tag or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in args.backup_tag):
        parser.error("backup tag must be a simple directory name")
    collect(args.root, args.root / "checkpoint_safety" / args.backup_tag, args.report)
