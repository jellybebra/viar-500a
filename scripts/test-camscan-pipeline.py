#!/usr/bin/env python3
"""Run the VIAR document pipeline on a saved camera frame without the GUI."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from camscan import postprocessing, scanner
import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--rotate",
        choices=(0, 90, 180, 270),
        default=270,
        type=int,
        help="Clockwise rotation; 270 is 90 degrees counter-clockwise",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.input))
    if image is None:
        raise SystemExit(f"Cannot read {args.input}")

    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if args.rotate:
        image = cv2.rotate(image, rotations[args.rotate])

    result = scanner.main(image)
    if result.warped is None:
        raise SystemExit("No document detected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preview = utils.draw_contour(image=image, contour=result.contour)
    corrected = postprocessing.document_color(result.warped)
    gray = postprocessing.document_gray(result.warped)
    black_and_white = postprocessing.black_and_white(result.warped)

    cv2.imwrite(str(args.output_dir / "01-preview.png"), preview)
    cv2.imwrite(str(args.output_dir / "02-cropped.png"), result.warped)
    cv2.imwrite(str(args.output_dir / "03-color-corrected.png"), corrected)
    cv2.imwrite(str(args.output_dir / "04-grayscale.png"), gray)
    cv2.imwrite(str(args.output_dir / "05-black-and-white.png"), black_and_white)

    rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(
        args.output_dir / "06-document.pdf",
        "PDF",
        resolution=300.0,
    )

    print(
        json.dumps(
            {
                "input_shape": list(image.shape),
                "contour": result.contour.tolist(),
                "output_shape": list(corrected.shape),
                "raw_channel_mean_bgr": np.mean(
                    result.warped.reshape(-1, 3), axis=0
                ).round(1).tolist(),
                "corrected_channel_mean_bgr": np.mean(
                    corrected.reshape(-1, 3), axis=0
                ).round(1).tolist(),
                "raw_channel_p90_bgr": np.percentile(
                    result.warped.reshape(-1, 3), 90, axis=0
                ).round(1).tolist(),
                "corrected_channel_p90_bgr": np.percentile(
                    corrected.reshape(-1, 3), 90, axis=0
                ).round(1).tolist(),
                "pdf": str(args.output_dir / "06-document.pdf"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
