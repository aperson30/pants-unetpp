"""Bounded public demo-asset audit; no model execution or archive extraction.

Inputs stay outside Git. Downloads are revision/size/SHA256 pinned. Public
dataset card says MIT; this is not certification of upstream clinical rights.
"""

import argparse
import hashlib
import gzip
import json
import re
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

REVISION = "b0516d354c4473ea4f8539cad395dfddb5944215"
ASSETS = {
    "Normal.tar": (6344704, "2c3342b79b57b457dd97fbbc1786ef56ce32f81d14e4b3e2caf91b7b03806f02"),
    "Demo.tar": (41293312, "eea5b165b14d5ae3b10212fe946466831a9edbbcbc8c236a58bb00997d8f6e47"),
}


def inspect(path):
    entries = []
    expanded = 0
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            parts = PurePosixPath(member.name)
            if parts.is_absolute() or ".." in parts.parts or "\\" in member.name:
                raise ValueError("Unsafe archive name")
            if not (member.isfile() or member.isdir()):
                raise ValueError("Unexpected link/device member")
            if member.size > 16*1024*1024:
                raise ValueError("Member exceeds audit size cap")
            expanded += member.size
            if expanded > 128*1024*1024 or len(entries) >= 512:
                raise ValueError("Archive exceeds audit caps")
            entries.append({"name": member.name, "bytes": member.size, "file": member.isfile()})
    return {"archive": path.name, "member_count": len(entries),
            "expanded_bytes": expanded, "members": entries}


def load_objects(root):
    import nibabel as nib
    import numpy as np

    objects = {}
    for name in ASSETS:
        size, digest = ASSETS[name]
        path = root/name
        if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Asset integrity mismatch")
        inspect(path)
        with tarfile.open(root/name, "r:") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".nii.gz"):
                    continue
                match = re.fullmatch(r"LIDC-IDRI-(\d+)_[A-Za-z]+_(\d+)\.nii\.gz", PurePosixPath(member.name).name)
                if not match:
                    raise ValueError("Unexpected pairing filename")
                group = str(PurePosixPath(member.name).parent)
                with archive.extractfile(member) as raw, gzip.GzipFile(fileobj=raw) as compressed:
                    payload = compressed.read(16*1024*1024+1)
                if len(payload) > 16*1024*1024:
                    raise ValueError("NIfTI exceeds decoded cap")
                image = nib.Nifti1Image.from_bytes(payload)
                if np.prod(image.shape) > 2*1024*1024:
                    raise ValueError("NIfTI shape exceeds cap")
                data = image.get_fdata(dtype=np.float32)
                if not np.isfinite(data).all():
                    raise ValueError("Nonfinite image values")
                key = (match[1], match[2])
                if (group, key) in objects:
                    raise ValueError("Duplicate pairing key")
                objects[group, key] = (data, image.affine, image.header.get_zooms())
    return objects


def geometry_audit(root):
    import numpy as np

    objects = load_objects(root)
    groups = sorted({group for group, _ in objects})
    counts = {group: sum(g == group for g, _ in objects) for group in groups}
    normal_keys = {key for group, key in objects if group == "Normal/Image"}
    rows = []
    for key in sorted(normal_keys):
        required = ["Normal/Image", "Normal/Mask"] + [f"Demo/{kind}/{kind}_{i}" for kind in ("Image", "Mask") for i in (1, 2, 3)]
        row = {"case_key_hash": hashlib.sha256((key[0]+":"+key[1]).encode()).hexdigest(),
               "missing_groups": [g for g in required if (g, key) not in objects]}
        if row["missing_groups"]:
            rows.append(row)
            continue
        normal, affine, spacing = objects["Normal/Image", key]
        mask = objects["Normal/Mask", key][0]
        row.update(normal_shape=list(normal.shape), normal_spacing=[float(v) for v in spacing],
                   normal_range=[float(normal.min()), float(normal.max())],
                   mask_values=np.unique(mask).tolist(), mask_positive_voxels=int((mask>0).sum()), variants=[])
        for i in (1, 2, 3):
            generated, ga, gs = objects[f"Demo/Image/Image_{i}", key]
            gm, ma, _ = objects[f"Demo/Mask/Mask_{i}", key]
            variant = {"variant": i, "shape": list(generated.shape), "spacing": [float(v) for v in gs],
                       "range": [float(generated.min()), float(generated.max())],
                       "same_grid": normal.shape == generated.shape == mask.shape == gm.shape and
                                    bool(np.allclose(affine, ga) and np.allclose(affine, ma) and
                                         np.allclose(affine, objects["Normal/Mask", key][1]))}
            if variant["same_grid"]:
                variant["same_mask"] = bool(np.array_equal(mask, gm))
                outside = gm <= 0
                inside = gm > 0
                # Pinned author loader: clamp HU, fixed rescaling to [-1,1].
                # The saved generated NIfTI is still normalized, NOT HU.
                normalized = (np.clip(normal, -1000, 400)+1000)/700-1
                residual = np.abs(generated[outside]-normalized[outside])
                variant["background_mae_normalized"] = float(residual.mean())
                variant["background_abs_error_quantiles_normalized"] = np.quantile(residual, [0.5, 0.95, 0.99, 1]).tolist()
                variant["foreground_mean_normalized"] = float(generated[inside].mean()) if inside.any() else None
                variant["foreground_mean_change_normalized"] = float((generated[inside]-normalized[inside]).mean()) if inside.any() else None
                variant["foreground_intensity_quantiles_normalized"] = np.quantile(generated[inside], [0.05, 0.5, 0.95]).tolist() if inside.any() else []
            row["variants"].append(variant)
        means = [v.get("foreground_mean_normalized") for v in row["variants"]]
        row["foreground_means_increasing_1_to_3"] = all(m is not None for m in means) and means[0] <= means[1] <= means[2]
        rows.append(row)
    return {"groups": counts, "normal_roi_count": len(normal_keys),
            "normal_unique_public_patient_ids": len({key[0] for key in normal_keys}),
            "all_cases": rows, "scope": "Array/geometry descriptors, not lesion realism, detection or independent clinical samples"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", required=True, type=Path)
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.asset_dir.resolve()
    if (root / ".git").exists() or "pants-unetpp-opt" in root.parts:
        raise ValueError("Raw data must be outside the repository")
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for name, (size, expected_hash) in ASSETS.items():
        target = root/name
        if not target.exists():
            url = f"https://huggingface.co/datasets/YuheLiuu/LeFusion_Preprocessed_Data/resolve/{REVISION}/LIDC-IDRI/{name}"
            with urllib.request.urlopen(url, timeout=45) as response, target.open("xb") as stream:
                total = 0
                while block := response.read(1024*1024):
                    total += len(block)
                    if total > size:
                        raise ValueError("Download exceeds advertised size; partial file retained")
                    stream.write(block)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if target.stat().st_size != size or digest != expected_hash:
            raise ValueError("Asset size/hash mismatch; no parsing performed")
        report = inspect(target)
        report["sha256"] = digest
        results.append(report)
    result = {"revision": REVISION, "gpu_hours": 0,
              "scope": "Integrity/member inventory only; upstream provenance and clinical adequacy unverified",
              "assets": [{k:v for k,v in r.items() if k != "members"} for r in results]}
    if args.geometry:
        result["geometry"] = geometry_audit(root)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "geometry"}, indent=2))
    if "geometry" in result:
        print(json.dumps({k:v for k,v in result["geometry"].items() if k != "all_cases"}, indent=2))


if __name__ == "__main__":
    main()
