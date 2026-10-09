"""Visual index-space label audit. No registration or clinical eligibility claim."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

from candidate_label_extension import ARCHIVE_SHA

EXPECTED = {"CV_00007899": 39444, "CV_00019311": 687140, "CV_00019743": 767050}


def main(root, archive_path):
    if (root / "mask_overlay.json").exists() or (root / "mask_overlay.png").exists():
        raise FileExistsError("Preserve prior output")
    digest = hashlib.sha256()
    with archive_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA:
        raise ValueError("Cached archive version differs")
    masks = {}
    # Buffer decompressed bytes, avoiding tarfile's compressed-stream buffering
    # regression on long runs of sparse masks. Reuse the verified cache only.
    with gzip.open(archive_path, "rb") as decoded_archive, tarfile.open(fileobj=decoded_archive, mode="r|", bufsize=65536) as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            cases = [part for part in path.parts if part in EXPECTED]
            if not cases or path.name != "pancreatic_lesion.nii.gz":
                continue
            if len(cases) != 1 or cases[0] in masks or not member.isfile() or member.size > 4 * 1024**2:
                raise ValueError("Malformed/duplicate mask member")
            with archive.extractfile(member) as source:
                raw = source.read(member.size + 1)
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as source:
                decoded = source.read(512 * 1024**2 + 1)
            if len(decoded) > 512 * 1024**2:
                raise ValueError("Decoded mask cap")
            img = nib.Nifti1Image.from_bytes(decoded)
            array = np.asanyarray(img.dataobj)
            if img.shape != (512, 512, 378) or not np.isfinite(array).all() or np.any(array < 0):
                raise ValueError("Unexpected target mask")
            if int(np.count_nonzero(array)) != EXPECTED[cases[0]]:
                raise ValueError("Mask foreground differs from previous audit")
            masks[cases[0]] = (array > 0, img.affine.copy(), img.header.get_xyzt_units(), hashlib.sha256(raw).hexdigest())
            destination = root / (cases[0] + "_public_pancreatic_lesion.nii.gz")
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != hashlib.sha256(raw).hexdigest():
                    raise ValueError("Existing isolated mask differs; preserve and stop")
            else:
                with destination.open("xb") as output:
                    output.write(raw)
            if set(masks) == set(EXPECTED):
                break
    if set(masks) != set(EXPECTED):
        raise ValueError("Not all target masks found")
    slices = (40, 84, 128, 155)
    canvas = Image.new("RGB", (1152, 410 * len(slices)), "black")
    draw = ImageDraw.Draw(canvas)
    results, planes = {}, {}
    for col, case in enumerate(EXPECTED):
        ct = nib.load(root / (case + ".nii.gz"))
        array = np.asanyarray(ct.dataobj)
        mask, affine, units, digest = masks[case]
        planes[case] = mask
        results[case] = {"mask_sha256": digest, "foreground_voxels": int(mask.sum()),
                         "mask_units": list(units), "mask_affine": affine.tolist(),
                         "ct_affine": ct.affine.tolist(),
                         "same_stored_affine": bool(np.allclose(affine, ct.affine, rtol=0, atol=1e-5)),
                         "mask_index_alignment_verified": False}
        for row, z in enumerate(slices):
            gray = (np.clip((array[:, :, z].astype(np.float32) + 160) / 400, 0, 1) * 255).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
            positive = mask[:, :, z]
            rgb[positive] = .5 * rgb[positive] + .5 * np.array([0, 255, 0])
            view = Image.fromarray(rgb.astype(np.uint8).transpose(1, 0, 2)[::-1]).resize((384, 384))
            canvas.paste(view, (col * 384, row * 410 + 25))
            draw.text((col * 384 + 6, row * 410 + 5), f"{case}, z={z}, index-space only", fill="white")
        del array
    canvas.save(root / "mask_overlay.png")
    comparisons = {}
    cases = list(EXPECTED)
    for i, a in enumerate(cases):
        for b in cases[i + 1:]:
            intersection = int(np.count_nonzero(planes[a] & planes[b]))
            comparisons[f"{a}:{b}"] = {"index_space_mask_dice": 2 * intersection / (EXPECTED[a] + EXPECTED[b]),
                                         "clinical_label_agreement_established": False}
    result = {"scope": "index-space overlays; source labels unchanged, no clinical/registration pass",
              "cases": results, "mask_comparisons": comparisons, "training_eligible": False, "gpu_hours": 0}
    with (root / "mask_overlay.json").open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    main(Path(args.root), Path(args.archive))
