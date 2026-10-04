"""CPU factorial-control contract on pinned public 3D demo arrays.

Not a quality metric, verifier experiment, clinical result or new method.
No generated images saved, no models loaded, no GPU or author-code execution.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from audit_lefusion_assets import REVISION, load_objects


def controls(source, generated, mask):
    """Return foreground-only and background-only interventions in common units."""
    if source.shape != generated.shape or mask.shape != source.shape:
        raise ValueError("Mismatched geometry")
    if not np.isin(mask, [0, 1]).all() or not (mask > 0).any() or (mask > 0).all():
        raise ValueError("Need binary, nonempty foreground and background")
    original = (np.clip(source, -1000, 400)+1000)/700-1
    inside = mask > 0
    foreground_only = np.where(inside, generated, original)
    background_only = np.where(inside, original, generated)
    assert np.array_equal(foreground_only[inside], generated[inside])
    assert np.array_equal(foreground_only[~inside], original[~inside])
    assert np.array_equal(background_only[inside], original[inside])
    assert np.array_equal(background_only[~inside], generated[~inside])
    np.testing.assert_allclose(foreground_only.astype(np.float64)+background_only,
                               generated.astype(np.float64)+original, rtol=0, atol=0)
    return original, foreground_only, background_only


def run(root):
    start = time.perf_counter()
    objects = load_objects(root)
    keys = sorted(key for group, key in objects if group == "Normal/Image")
    rows = []
    for key in keys:
        source, affine, _ = objects["Normal/Image", key]
        mask, mask_affine, _ = objects["Normal/Mask", key]
        np.testing.assert_allclose(affine, mask_affine)
        for variant in (1, 2, 3):
            generated, ga, _ = objects[f"Demo/Image/Image_{variant}", key]
            generated_mask, ma, _ = objects[f"Demo/Mask/Mask_{variant}", key]
            np.testing.assert_allclose(affine, ga)
            np.testing.assert_allclose(affine, ma)
            assert np.array_equal(mask, generated_mask)
            original, foreground, background = controls(source, generated, mask)
            total_energy = float(np.square(generated.astype(np.float64)-original).sum())
            fg_energy = float(np.square(foreground.astype(np.float64)-original).sum())
            bg_energy = float(np.square(background.astype(np.float64)-original).sum())
            np.testing.assert_allclose(fg_energy+bg_energy, total_energy, rtol=1e-12, atol=1e-12)
            rows.append({"case_key_hash": hashlib.sha256((key[0]+":"+key[1]).encode()).hexdigest(),
                "variant": variant, "foreground_voxels": int((mask>0).sum()),
                "squared_change_sum": total_energy, "foreground_squared_change_sum": fg_energy,
                "background_squared_change_sum": bg_energy,
                "background_fraction_of_squared_change": bg_energy/total_energy if total_energy else None})
    fractions = [r["background_fraction_of_squared_change"] for r in rows]
    return {"revision": REVISION, "gpu_hours": 0, "cpu_seconds": time.perf_counter()-start,
        "source_rois": len(keys), "variant_pairs": len(rows),
        "all_counterfactual_and_decomposition_assertions_passed": True,
        "background_fraction_min_median_max": np.quantile(fractions, [0, 0.5, 1]).tolist(),
        "scope": "Squared intensity change relative to source, NOT synthesis error, quality or task harm",
        "warning": "Hard composites may introduce boundary seams; model response is not automatically shortcut evidence",
        "all_pairs": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.asset_dir)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "all_pairs"}, indent=2))
