"""CPU preparation for candidate-paired architecture feasibility, not medical proof.

Pin four public files; retain independent metadata groups as provisional splits.
No training eligibility certification, registration, augmentation or protected data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

from candidate_pixel_audit import REV, stage

FILES = {
    "CV_00018276": (118253758, "c668a4798714530c1f0bec09b1c0c1961252dee744df4d663ce2fc775d31abba"),
    "CV_00008477": (115815466, "595980567a1ce42d41cc88d8f2d7ac20aca06e0a214ecd96ee0876cef3098eb4"),
    "CV_00011916": (108881118, "9ec0d7ef41560429228df6ea1149b6f5e4c11a096c99a53430c9fc183424c8cc"),
    "CV_00018699": (109091800, "2a8639064021a0e076976934df92037f3a730a7708b2408564e30655225a34d9"),
}
PAIRS = {"train": ("CV_00018276", "CV_00008477", "venous"),
         "development": ("CV_00011916", "CV_00018699", "arterial")}
STARTS = (84, 120, 156, 192)


def triplets(array):
    if array.ndim != 3 or array.shape[:2] != (512, 512) or array.shape[2] < max(STARTS) + 3:
        raise ValueError("Unexpected CT dimensions")
    values = np.stack([np.moveaxis(array[:, :, z:z+3], -1, 0) for z in STARTS]).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite CT triplet")
    # Explicit baseline [-1000,1000] clip then [-1,1], no spatial resize or registration.
    return np.clip(values, -1000, 1000) / 1000


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in ("pilot_data.json", "triplets.npz", "candidate_pairs.png")):
        raise FileExistsError("Preserve prepared outputs; use a fresh directory")
    if sum(size for size, _ in FILES.values()) > 500_000_000:
        raise ValueError("Public CT staging exceeds 500MB cap")
    deadline = time.monotonic() + 300
    data, records, affines = {}, {}, {}
    for split, (source, target, phase) in PAIRS.items():
        for role, case in (("source", source), ("target", target)):
            path = stage(root, case, deadline, FILES)
            image = nib.load(path)
            if (image.header.get_xyzt_units()[0] != "mm" or not np.isfinite(image.affine).all()
                    or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12
                    or np.prod(image.shape) * image.get_data_dtype().itemsize > 512 * 1024**2):
                raise ValueError("Unexpected CT geometry/size")
            pixels = np.asanyarray(image.dataobj)
            if pixels.nbytes > 1024**3 or not np.isfinite(pixels).all():
                raise ValueError("Decoded volume cap or nonfinite voxels")
            data[f"{split}_{role}"] = triplets(pixels)
            affines[case] = image.affine.tolist()
            records[case] = {"file_sha256": FILES[case][1], "shape": list(image.shape),
                             "spacing": list(map(float, image.header.get_zooms()[:3])),
                             "scaled_voxel_sha256": hashlib.sha256(pixels.tobytes(order="F")).hexdigest()}
            del pixels
        if records[source]["scaled_voxel_sha256"] == records[target]["scaled_voxel_sha256"]:
            raise ValueError("Source and target are duplicate voxel volumes")
    with (root / "triplets.npz").open("xb") as output:
        np.savez_compressed(output, **data)
    digest = hashlib.sha256((root / "triplets.npz").read_bytes()).hexdigest()
    canvas = Image.new("RGB", (384*4, 410*len(STARTS)), "black")
    draw = ImageDraw.Draw(canvas)
    for col, key in enumerate(data):
        for row, z in enumerate(STARTS):
            hu = data[key][row, 1] * 1000
            gray = (np.clip((hu + 160) / 400, 0, 1) * 255).astype(np.uint8)
            canvas.paste(Image.fromarray(gray.T[::-1]).resize((384, 384)), (col*384, row*410+25))
            draw.text((col*384+6, row*410+5), f"{key}, z={z+1}; candidate only", fill="white")
    canvas.save(root / "candidate_pairs.png")
    result = {"scope": "candidate-paired engineering inputs, not medical or statistical evaluation",
              "revision": REV, "files": records, "affines": affines,
              "provisional_pairs": {split: {"source": source, "target": target, "phase_hint": phase}
                                    for split, (source, target, phase) in PAIRS.items()},
              "triplet_starts": list(STARTS), "arrays": {key: list(value.shape) for key, value in data.items()},
              "triplets_sha256": digest, "registration_applied": False,
              "clinical_phase_verified": False, "patient_independence_certified": False,
              "protected_PanTS_overlap_excluded": False, "training_eligible": False,
              "gpu_hours": 0}
    with (root / "pilot_data.json").open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    main(Path(parser.parse_args().out_dir))
