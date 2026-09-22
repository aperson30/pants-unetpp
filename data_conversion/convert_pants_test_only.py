"""Convert only PanTS-te, avoiding a second 9,000-case training-data staging pass."""
from __future__ import annotations

import argparse
from pathlib import Path

from data_conversion.convert_pants_to_nnunet import convert_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pants-root", type=Path, required=True)
    parser.add_argument("--images-ts-dir", type=Path, required=True)
    parser.add_argument("--test-answer-key-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=901)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    case_ids = convert_split(
        args.pants_root,
        "Te",
        args.images_ts_dir,
        args.test_answer_key_dir,
        resume=args.resume,
    )
    if len(case_ids) != args.expected_cases:
        raise RuntimeError(f"expected {args.expected_cases} test cases, found {len(case_ids)}")
    print(f"verified {len(case_ids)} PanTS-te cases")


if __name__ == "__main__":
    main()
