"""
This module provides utility postprocessing functions for images.
"""

import cv2
import numpy as np


def dummy(image: cv2.Mat) -> cv2.Mat:
    """
    Apply no processing whatsoever and simply return the image again.
    :param image: The input image
    :return: The original image with no modification
    """
    return image


def sharpen(image: cv2.Mat) -> cv2.Mat:
    """
    Apply a sharpening effect to the input image.
    :param image: The input image
    :return: The image with the effect applied
    """
    blurred = cv2.GaussianBlur(
        src=image,
        ksize=(0, 0),
        sigmaX=3,
    )
    sharpened = cv2.addWeighted(
        src1=image,
        alpha=1.5,
        src2=blurred,
        beta=-0.5,
        gamma=0,
    )
    return sharpened


def grayscale(image: cv2.Mat) -> cv2.Mat:
    """
    Convert the input image to grayscale.
    :param image: The input image
    :return: The image with the effect applied
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _normalize_percentiles(channel: cv2.Mat, low: float = 1, high: float = 99):
    """Stretch a channel robustly without letting a few outliers set the range."""
    lo, hi = np.percentile(channel, (low, high))
    if hi <= lo + 1:
        return channel.copy()
    scaled = (channel.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def document_color(image: cv2.Mat) -> cv2.Mat:
    """
    Correct the strong green/cyan cast produced by the VIAR-500A.

    The brightest pixels in a document scan normally belong to the paper.  We
    use those pixels as a neutral-white reference, then improve local contrast
    in the luminance channel.  Unlike fixed RGB gains this also adapts when the
    lamp, ambient light, or paper changes.
    """
    if image is None or image.size == 0:
        return image
    if len(image.shape) == 2:
        return document_gray(image)

    # Correct both the camera's coloured black level and its unequal channel
    # gains.  Mapping each channel's paper reference to the same value removes
    # the VIAR's cyan cast much more reliably than a simple gray-world gain.
    pixels = image.reshape(-1, 3)
    black_reference = np.percentile(pixels, 1, axis=0)
    paper_percentile = 90 if pixels.shape[0] > 10_000 else 95
    paper_reference = np.percentile(pixels, paper_percentile, axis=0)
    corrected_channels = []
    for channel_index, channel in enumerate(cv2.split(image)):
        black = float(black_reference[channel_index])
        paper = float(paper_reference[channel_index])
        if paper <= black + 1:
            corrected_channels.append(channel)
            continue
        normalized = (
            (channel.astype(np.float32) - black)
            * (240.0 / (paper - black))
            + 5.0
        )
        corrected_channels.append(np.clip(normalized, 0, 255).astype(np.uint8))

    corrected = cv2.merge(corrected_channels)
    # Illumination across the VIAR bed is not linear: after black/white point
    # correction the shadowed parts of white paper can still look cyan.  Reduce
    # chroma only for bright, weakly saturated pixels (paper), preserving truly
    # coloured stamps, signatures, and photographs.
    hsv = cv2.cvtColor(corrected, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    value = hsv[:, :, 2].astype(np.float32)
    neutral_weight = np.clip((value - 45.0) / 150.0, 0.0, 1.0)
    neutral_weight *= np.clip((115.0 - saturation) / 85.0, 0.0, 1.0)
    neutral_weight *= 0.92
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB).astype(np.float32)
    for index in (1, 2):
        lab[:, :, index] = (
            128.0 + (lab[:, :, index] - 128.0) * (1.0 - neutral_weight)
        )
    corrected = cv2.cvtColor(
        np.clip(lab, 0, 255).astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )
    blurred = cv2.GaussianBlur(corrected, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(corrected, 1.12, blurred, -0.12, 0)


def document_gray(image: cv2.Mat) -> cv2.Mat:
    """Create a neutral, high-contrast grayscale document image."""
    gray = image if len(image.shape) == 2 else grayscale(image)
    gray = _normalize_percentiles(gray, low=1, high=99)
    return cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)


def black_and_white(
    image: cv2.Mat,
    threshold_strength: int = 50,
) -> cv2.Mat:
    """
    Convert a document to black and white using local Sauvola thresholds.

    Each part of the page gets its own threshold, so faint text remains visible
    even when the page is lit unevenly.  ``threshold_strength`` controls how
    readily gray pixels are retained as ink.
    :param image: The input image
    :param threshold_strength: Ink-retention strength from 0 to 100
    :return: The image with the effect applied
    """
    gray = image if len(image.shape) == 2 else grayscale(image)

    short_side = min(gray.shape[:2])
    block_size = max(31, int(short_side * 0.025))
    block_size |= 1

    strength = np.clip(threshold_strength, 0, 100)
    # In Sauvola, a smaller k raises the local threshold and therefore retains
    # more faint strokes as black pixels.
    k = float(np.interp(strength, [0, 100], [0.35, 0.05]))

    return cv2.ximgproc.niBlackThreshold(
        gray,
        maxValue=255,
        type=cv2.THRESH_BINARY,
        blockSize=block_size,
        k=k,
        binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA,
        r=128,
    )
