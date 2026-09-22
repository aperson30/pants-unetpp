"""Convert only PanTS training data for a node-local training allocation."""
from __future__ import annotations

import argparse
from pathlib import Path

from data_conversion.convert_pants_to_nnunet import convert_split, write_dataset_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pants-root", type=Path, required=True)
    parser.add_argument("--nnunet-dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=9000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    case_ids = convert_split(
        args.pants_root,
        "Tr",
        args.nnunet_dataset_dir / "imagesTr",
        args.nnunet_dataset_dir / "labelsTr",
        resume=args.resume,
    )
    if len(case_ids) != args.expected_cases:
        raise RuntimeError(f"expected {args.expected_cases} training cases, found {len(case_ids)}")
    write_dataset_json(args.nnunet_dataset_dir, num_training=len(case_ids))
    print(f"verified {len(case_ids)} PanTS training cases")


if __name__ == "__main__":
    main()
