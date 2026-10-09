"""Isolated, bounded public candidate audit, not clinical phase certification.

Three pinned scans, <=300MB compressed total. Never edit source files or labels.
No model training, patient reports, automatic registration or test evaluation.
"""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

REV = "c15802c28cb31b71dee2e680319d522b3bc4cbbf"
FILES = {
    "CV_00007899": (71769950, "c11802be029940992df22fad504100ee58dceccea953ac154465a9755ed7d201"),
    "CV_00019311": (102890209, "728c47738ad9d0d8c8fb7283d9c07317d5546ce443880a4068a3614a29998e0a"),
    "CV_00019743": (111484203, "784fe634145a8355f6cf973e8d3b826c7ccdc4799200902baf5fff5f02ca3a42"),
}


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def stage(root, case, deadline):
    size, digest = FILES[case]
    path = root / f"{case}.nii.gz"
    if path.exists():
        if path.stat().st_size != size or file_hash(path) != digest:
            raise ValueError("Existing candidate differs; preserve and stop")
        return path
    partial = root / f"{case}.partial"
    with urllib.request.urlopen(
        f"https://huggingface.co/datasets/BodyMaps/CancerVerse/resolve/{REV}/CancerVerse/{case}/ct.nii.gz",
        timeout=25,
    ) as source, partial.open("xb") as output:
        count = 0
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("Candidate staging deadline; no retry")
            chunk = source.read(min(1024 * 1024, size - count + 1))
            if not chunk:
                break
            count += len(chunk)
            if count > size:
                raise ValueError("Download exceeded pinned length")
            output.write(chunk)
    if count != size or file_hash(partial) != digest:
        raise ValueError("Pinned CT integrity mismatch; preserve partial and stop")
    partial.rename(path)
    print(json.dumps({"staged": case, "bytes": size}), flush=True)
    return path


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "pixel_audit.json").exists() or (root / "slices.png").exists():
        raise FileExistsError("Use a fresh output directory")
    if sum(size for size, _ in FILES.values()) > 300_000_000:
        raise ValueError("Compressed download cap")
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise RuntimeError("Need 2GiB free before isolated audit")
    deadline = time.monotonic() + 480
    sampled, geometries, slice_images, records = {}, {}, {}, {}
    slices = (40, 84, 128, 155, 210, 290)
    for case in FILES:
        path = stage(root, case, deadline)
        img = nib.load(path)
        if img.shape != (512, 512, 378) or np.prod(img.shape) * img.get_data_dtype().itemsize > 512 * 1024**2:
            raise ValueError("Unexpected shape or decoded size")
        if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-12:
            raise ValueError("Invalid image affine")
        # Materialize once: repeated gzip slices would otherwise repeatedly decompress.
        array = np.asanyarray(img.dataobj)
        if array.nbytes > 1024**3 or not np.isfinite(array).all():
            raise ValueError("Invalid/beyond-cap scaled volume")
        tiny = np.asarray(array[::4, ::4, ::4], dtype=np.float32).copy()
        sampled[case] = tiny
        geometries[case] = img.affine.tolist()
        records[case] = {"shape": list(img.shape), "file_sha256": FILES[case][1],
                         "scaled_voxel_sha256": hashlib.sha256(array.tobytes(order="F")).hexdigest(),
                         "voxel_min": float(array.min()), "voxel_max": float(array.max()),
                         "sampled_percentiles": np.percentile(tiny, [1, 50, 99]).tolist()}
        slice_images[case] = []
        for z in slices:
            gray = (np.clip((array[:, :, z].astype(np.float32) + 160) / 400, 0, 1) * 255).astype(np.uint8)
            slice_images[case].append(Image.fromarray(gray.T[::-1]).resize((384, 384)))
        del array, img
    correlations = {}
    cases = list(FILES)
    for index, a in enumerate(cases):
        for b in cases[index + 1:]:
            x, y = sampled[a], sampled[b]
            # Descriptive only: unregistered, subsampled, and not tumor-specific.
            body = (x > -500) & (y > -500) & (x < 500) & (y < 500)
            correlations[f"{a}:{b}"] = {
                "sampled_voxels_equal": bool(np.array_equal(x, y)),
                "sampled_body_voxels": int(body.sum()),
                "unregistered_body_correlation": float(np.corrcoef(x[body], y[body])[0, 1]),
                "unregistered_body_mean_abs_difference": float(np.abs(x[body] - y[body]).mean()),
            }
    canvas = Image.new("RGB", (384 * 3, 410 * len(slices)), "black")
    draw = ImageDraw.Draw(canvas)
    for col, case in enumerate(cases):
        for row, z in enumerate(slices):
            canvas.paste(slice_images[case][row], (col * 384, row * 410 + 25))
            draw.text((col * 384 + 8, row * 410 + 5), f"{case}, z={z}, window [-160,240]", fill="white")
    canvas.save(root / "slices.png")
    result = {"scope": "public candidate pixels; no clinical or registration pass",
              "revision": REV, "files": records, "affines": geometries,
              "descriptive_pairs": correlations, "training_eligible": False,
              "tumor_labels_inspected_in_this_run": False, "gpu_hours": 0}
    with (root / "pixel_audit.json").open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    main(Path(parser.parse_args().out_dir))
