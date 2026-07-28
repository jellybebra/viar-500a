#!/usr/bin/env python3
"""Create UI-equivalent black-and-white strength variants for visual tuning."""

import argparse
from pathlib import Path

import cv2

from camscan import postprocessing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    image = cv2.imread(str(args.input))
    if image is None:
        raise SystemExit(f"Cannot read {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for strength in (0, 35, 70, 100):
        output = postprocessing.black_and_white(
            image,
            threshold_strength=strength,
        )
        cv2.imwrite(
            str(args.output_dir / f"bw-strength-{strength}.png"),
            output,
        )


if __name__ == "__main__":
    main()
