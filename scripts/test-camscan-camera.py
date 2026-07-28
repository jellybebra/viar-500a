#!/usr/bin/env python3
"""Headless smoke test for the live VIAR-500A camera pipeline."""

import argparse
import time
from pathlib import Path

import cv2

from camscan.camera import Camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--warmup", type=float, default=6.0)
    args = parser.parse_args()

    camera = Camera(warmup_seconds=args.warmup)
    try:
        deadline = time.monotonic() + args.warmup + 5
        while time.monotonic() < deadline and not camera.is_warmed_up:
            time.sleep(0.2)
        image = camera.capture()
        if image is None:
            raise SystemExit("Camera returned no frame")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), image):
            raise SystemExit(f"Could not write {args.output}")
        print(
            f"captured={image.shape[1]}x{image.shape[0]} "
            f"warmed_up={camera.is_warmed_up} output={args.output}"
        )
    finally:
        camera.close()


if __name__ == "__main__":
    main()
