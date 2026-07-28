"""Tests for document postprocessing."""

import cv2
import numpy as np

from camscan import postprocessing


def test_black_and_white_sauvola_preserves_more_ink_at_high_strength():
    illumination = np.linspace(185, 245, 320, dtype=np.uint8)
    page_gray = np.repeat(illumination[np.newaxis, :], 240, axis=0)
    page = cv2.cvtColor(page_gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(
        page,
        "Faint text",
        (25, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (150, 150, 150),
        2,
    )

    low = postprocessing.black_and_white(page, threshold_strength=0)
    high = postprocessing.black_and_white(page, threshold_strength=100)

    assert low.shape == page.shape[:2]
    assert set(np.unique(low)).issubset({0, 255})
    assert np.count_nonzero(high == 0) > np.count_nonzero(low == 0)
