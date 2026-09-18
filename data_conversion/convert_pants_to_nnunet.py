"""
Converts PanTS's raw downloaded format into the folder/file layout nnU-Net v2 requires.

Run this once, after `download_PanTS_data.sh` and `download_PanTS_label.sh` (from the PanTS repo)
have finished downloading, and before any nnU-Net preprocessing/training commands.
"""
from pathlib import Path
import json
import shutil

import nibabel as nib
import numpy as np

# From PanTS/data/README.md -- maps each organ's file name to the class number nnU-Net should learn.
CLASS_MAP = {
    'adrenal_gland_left': 1, 'adrenal_gland_right': 2, 'aorta': 3, 'bladder': 4,
    'celiac_artery': 5, 'colon': 6, 'common_bile_duct': 7, 'duodenum': 8,
    'femur_left': 9, 'femur_right': 10, 'gall_bladder': 11, 'kidney_left': 12,
    'kidney_right': 13, 'liver': 14, 'lung_left': 15, 'lung_right': 16,
    'pancreas': 17, 'pancreas_body': 18, 'pancreas_head': 19, 'pancreas_tail': 20,
    'pancreatic_duct': 21, 'postcava': 22, 'prostate': 23, 'spleen': 24,
    'stomach': 25, 'superior_mesenteric_artery': 26, 'veins': 27, 'pancreatic_lesion': 28,
}


def merge_labels(segmentations_folder: Path, class_map: dict) -> tuple:
    """Combine one case's 28 separate yes/no organ files into a single labeled volume."""
    combined = None
    reference_img = None

    # tumor last, so it always overwrites the surrounding pancreas label on any overlap
    ordered_organs = [name for name in class_map if name != "pancreatic_lesion"]
    ordered_organs.append("pancreatic_lesion")

    for organ_name in ordered_organs:
        file_path = segmentations_folder / f"{organ_name}.nii.gz"
        if not file_path.is_file():
            continue

        img = nib.load(str(file_path))
        mask = np.asanyarray(img.dataobj) > 0

        if combined is None:
            combined = np.zeros(mask.shape, dtype=np.uint8)
            reference_img = img

        class_id = class_map[organ_name]
        combined[mask] = class_id

    return combined, reference_img


def process_case(case_id: str, ct_path: Path, segmentations_folder: Path,
                  out_images_dir: Path, out_labels_dir: Path) -> None:
    """Handle one case: copy its CT scan over, and merge+save its combined label volume."""
    out_image_path = out_images_dir / f"{case_id}_0000.nii.gz"
    shutil.copy(ct_path, out_image_path)

    combined, reference_img = merge_labels(segmentations_folder, CLASS_MAP)
    out_label_path = out_labels_dir / f"{case_id}.nii.gz"
    nib.save(nib.Nifti1Image(combined, reference_img.affine, reference_img.header), str(out_label_path))


def convert_split(pants_root: Path, split: str, out_images_dir: Path, out_labels_dir: Path,
                   limit: int = None, resume: bool = False) -> list:
    """split is 'Tr' (train) or 'Te' (test). Returns the list of case IDs processed (all of them,
    including any skipped because they were already done -- callers need the full list for
    numTraining/numTest, not just what ran this invocation).
    limit: if set, only process the first `limit` cases -- for a quick real-data sanity check
    before committing to the full 9,901-case run.
    resume: if set, skip any case whose image AND label output already exist -- for recovering
    from an interrupted run (crash, OOM, a concurrent job on the same node starving it) without
    redoing already-converted cases. A case with only ONE of the two outputs present (e.g. the
    label write was interrupted mid-case) is treated as NOT done and gets reprocessed --
    process_case()'s shutil.copy + nib.save are both idempotent overwrites, so this is safe."""
    image_root = pants_root / f"Image{split}"
    label_root = pants_root / f"Label{split}"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    case_ids = sorted(p.name for p in image_root.iterdir() if p.is_dir())
    if limit is not None:
        case_ids = case_ids[:limit]

    todo = case_ids
    if resume:
        todo = [c for c in case_ids
                if not ((out_images_dir / f"{c}_0000.nii.gz").is_file()
                        and (out_labels_dir / f"{c}.nii.gz").is_file())]
        skipped = len(case_ids) - len(todo)
        if skipped:
            print(f"resume: skipping {skipped}/{len(case_ids)} already-converted cases")

    for i, case_id in enumerate(todo, start=1):
        ct_path = image_root / case_id / "ct.nii.gz"
        segmentations_folder = label_root / case_id / "segmentations"
        process_case(case_id, ct_path, segmentations_folder, out_images_dir, out_labels_dir)
        print(f"[{i}/{len(todo)}] done: {case_id}")

    return case_ids


def write_dataset_json(nnunet_dataset_dir: Path, num_training: int) -> None:
    """Writes the small fact-sheet nnU-Net needs alongside the data."""
    labels = {"background": 0}
    labels.update(CLASS_MAP)

    dataset_json = {
        "channel_names": {"0": "CT"},   # which kind of scan channel "_0000" is -- see the quiz earlier
        "labels": labels,               # category name -> number, background must be spelled out too
        "numTraining": num_training,    # how many training scans there are
        "file_ending": ".nii.gz",       # the file format all our scans/labels are saved in
    }

    with open(nnunet_dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pants-root", type=Path, required=True,
                         help="path to the folder where you downloaded PanTS")
    parser.add_argument("--nnunet-dataset-dir", type=Path, required=True,
                         help="e.g. .../nnUNet_raw/Dataset001_PanTS -- where nnU-Net-ready files go")
    parser.add_argument("--test-answer-key-dir", type=Path, required=True,
                         help="separate folder to keep the true test-set labels, for grading later")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N cases per split -- use this for a quick "
                              "sanity check on real data before running the full conversion")
    parser.add_argument("--resume", action="store_true",
                         help="skip cases whose image+label output already exist -- for recovering "
                              "an interrupted run without redoing completed cases")
    args = parser.parse_args()

    train_ids = convert_split(
        args.pants_root, "Tr",
        args.nnunet_dataset_dir / "imagesTr", args.nnunet_dataset_dir / "labelsTr",
        limit=args.limit, resume=args.resume
    )
    test_ids = convert_split(
        args.pants_root, "Te",
        args.nnunet_dataset_dir / "imagesTs", args.test_answer_key_dir,
        limit=args.limit, resume=args.resume
    )

    write_dataset_json(args.nnunet_dataset_dir, num_training=len(train_ids))
    print(f"\nDone. {len(train_ids)} training cases, {len(test_ids)} test cases converted.")
