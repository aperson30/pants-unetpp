"""Assemble three metadata-distinct candidate groups for engineering only.

Train contains both venous/arterial hints; development uses a separate venous
candidate group. Avoid an entirely unseen development phase in a tiny pilot.
No certified patient independence, tumor ground truth or protected-set exclusion.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

V1_HASH = "87220c9eca7bb8c0dfd29a896676b60f61f2d730f4a05a4ecea6afe45d63f91e"
DEV = {"CV_00007899": "c11802be029940992df22fad504100ee58dceccea953ac154465a9755ed7d201",
       "CV_00019311": "728c47738ad9d0d8c8fb7283d9c07317d5546ce443880a4068a3614a29998e0a"}


def combine_arrays(first, second, development):
    arrays = list(first) + list(second) + list(development)
    if len(arrays) != 6:
        raise ValueError("Need three source/target pairs")
    for array in arrays:
        if array.shape != (4, 3, 512, 512) or array.dtype != np.float32 or not np.isfinite(array).all() or np.max(np.abs(array)) > 1:
            raise ValueError("Unexpected normalized CT triplets")
    return {"train_source": np.concatenate((first[0], second[0])),
            "train_target": np.concatenate((first[1], second[1])),
            "train_phase_index": np.array([0] * 4 + [1] * 4, dtype=np.int64),
            "development_source": development[0], "development_target": development[1],
            "development_phase_index": np.zeros(4, dtype=np.int64)}


def main(v1, dev_root, output):
    from architecture_pilot_data import triplets
    from candidate_pixel_audit import file_hash
    import nibabel as nib
    output.mkdir(parents=True, exist_ok=True)
    if (output / "triplets.npz").exists() or (output / "pilot_data.json").exists():
        raise FileExistsError("Use fresh output directory")
    if file_hash(v1 / "triplets.npz") != V1_HASH:
        raise ValueError("Prepared input hash differs")
    with np.load(v1 / "triplets.npz", allow_pickle=False) as dataset:
        first = (dataset["train_source"], dataset["train_target"])
        second = (dataset["development_source"], dataset["development_target"])
    dev_arrays = []
    for case, digest in DEV.items():
        path = dev_root / f"{case}.nii.gz"
        if file_hash(path) != digest:
            raise ValueError("Cached development CT differs")
        image = nib.load(path)
        dev_arrays.append(triplets(np.asanyarray(image.dataobj)))
    combined = combine_arrays(first, second, dev_arrays)
    with (output / "triplets.npz").open("xb") as stream:
        np.savez_compressed(stream, **combined)
    result = {"scope": "candidate-paired architecture-only exploratory feasibility",
              "triplets_sha256": file_hash(output / "triplets.npz"),
              "source_v1_triplets_sha256": V1_HASH, "development_ct_hashes": DEV,
              "training_case_ids": ["CV_00018276", "CV_00008477", "CV_00011916", "CV_00018699"],
              "development_case_ids": list(DEV), "phase_prompts": ["An venous phase CT slice.", "An arterial phase CT slice."],
              "normalized_range": [-1, 1], "shape": {key: list(value.shape) for key, value in combined.items()},
              "clinical_phase_verified": False, "patient_independence_certified": False,
              "protected_PanTS_overlap_excluded": False, "clinical_training_eligible": False,
              "registration_applied": False, "medical_quality_tested": False, "gpu_hours": 0}
    with (output / "pilot_data.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", required=True)
    parser.add_argument("--development-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    main(Path(args.v1), Path(args.development_root), Path(args.out_dir))
