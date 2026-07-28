#!/usr/bin/env python3
"""Benchmark one background preview pass without creating a Tk window."""

import argparse
import time

import cv2

from camscan.app import CamScanApp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"Cannot read {args.input}")

    for option in ("Без обработки", "Чёрно-белый"):
        started = time.perf_counter()
        preview, _, warped, contour = CamScanApp._process_preview_frame(
            image,
            option,
            20,
            False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        print(
            f"option={option} elapsed_ms={elapsed_ms:.1f} "
            f"preview={preview.shape[1]}x{preview.shape[0]} "
            f"document={warped is not None} contour={contour is not None}"
        )


if __name__ == "__main__":
    main()
