"""Tests for the user-facing page-size extraction modes."""

import cv2
import numpy as np

from camscan import app


def document_scene() -> np.ndarray:
    image = np.full((900, 1200, 3), 15, dtype=np.uint8)
    cv2.fillConvexPoly(
        image,
        np.array([[75, 110], [545, 85], [555, 810], [90, 825]]),
        (238, 238, 238),
    )
    cv2.fillConvexPoly(
        image,
        np.array([[650, 90], [1125, 115], [1105, 820], [635, 795]]),
        (245, 245, 245),
    )
    return image


def test_multiple_objects_produces_one_page_per_sheet():
    pages, contours = app.extract_pages(
        document_scene(),
        scan_size=app.SCAN_SIZE_MULTIPLE,
        custom_rectangle=(0.1, 0.1, 0.9, 0.9),
    )

    assert len(pages) == 2
    assert len(contours) == 2
    assert all(page.size > 0 for page in pages)


def test_custom_mode_uses_selected_normalized_rectangle():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    pages, contours = app.extract_pages(
        image,
        scan_size=app.SCAN_SIZE_CUSTOM,
        custom_rectangle=(0.25, 0.20, 0.75, 0.80),
    )

    assert len(pages) == 1
    assert pages[0].shape[:2] == (60, 100)
    np.testing.assert_array_equal(
        contours[0],
        np.array([[50, 20], [149, 20], [149, 79], [50, 79]]),
    )


def test_fixed_a4_crops_the_center_half_of_the_a3_work_area():
    image = np.zeros((900, 1200, 3), dtype=np.uint8)
    pages, contours = app.extract_pages(
        image,
        scan_size=app.SCAN_SIZE_A4,
        custom_rectangle=(0, 0, 1, 1),
    )

    assert len(pages) == 1
    page_height, page_width = pages[0].shape[:2]
    np.testing.assert_allclose(
        page_width / page_height,
        210 / 297,
        atol=0.01,
    )
    np.testing.assert_allclose(np.mean(contours[0], axis=0), [600, 450], atol=1)
