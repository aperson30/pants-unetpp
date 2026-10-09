"""Read-only public demo inspection. No protected cases, masks, or output CT edits."""
import argparse
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from baseline_contracts import validate_geometry


def main(path):
    path = Path(path).resolve(strict=True)
    image = nib.load(path)
    validate_geometry(tuple(int(n) for n in image.shape), torch.tensor(image.affine))
    if image.shape[2] < 3:
        raise ValueError("Need at least three slices")
    start = (image.shape[2] - 3) // 2
    triplet = np.asanyarray(image.dataobj[:, :, start:start + 3])
    if not np.isfinite(triplet).all():
        raise ValueError("Nonfinite source demo")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    print(json.dumps({"scope": "public demo source metadata and center triplet only",
        "path": str(path), "sha256": digest.hexdigest(), "shape_hwd": list(image.shape),
        "affine": image.affine.tolist(), "spacing": list(map(float, image.header.get_zooms())),
        "center_triplet_start": start, "center_min": float(triplet.min()),
        "center_max": float(triplet.max()), "paired_registration_verified": False,
        "tumor_labels_available": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    main(parser.parse_args().path)
