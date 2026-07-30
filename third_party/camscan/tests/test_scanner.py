"""
This module tests the functionality of the scanner code.
"""

from __future__ import annotations

import cv2
import math
import pytest
import numpy as np

from camscan import scanner


@pytest.mark.parametrize(
    "graph, length, expected_cycles",
    [
        [
            # Graph with 5 nodes and 6 edges
            [[1, 3], [0, 2, 4], [1, 3], [2, 4], [1, 3]],
            4,
            [
                [0, 1, 2, 3],
                [0, 1, 4, 3],
                [1, 2, 3, 4],
            ],
        ],
        [
            # Fully connected graph with 4 nodes and 8 edges
            # Here, all cycles will traverse the same nodes in different ways
            [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
            4,
            [
                [0, 1, 3, 2],
                [0, 2, 1, 3],
                [0, 1, 2, 3],
            ],
        ],
    ],
)
def test_find_cycles(
    graph: list[set[int]],
    length: int,
    expected_cycles: list[list[int]],
):
    """
    Test the functionality of the cycle finder on some test graphs.
    """
    actual_cycles = scanner.find_cycles(graph=graph, length=length)
    actual_cycles = list(sorted(map(tuple, actual_cycles)))
    expected_cycles = list(sorted(map(tuple, expected_cycles)))
    assert actual_cycles == expected_cycles


@pytest.mark.parametrize(
    "contour, expected_contour",
    [
        [
            # The input contour is on the form (TL, BL, BR, TR)
            # The ordered contour should be as (TL, TR, BR, BL)
            np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        ],
        [
            # The input contour is on the form (TL, BL, BR, TR)
            # The ordered contour should be as (TL, TR, BR, BL)
            np.array([[0, 1], [1, 2], [2, 1], [1, 0]]),
            np.array([[0, 1], [1, 0], [2, 1], [1, 2]]),
        ],
        [
            # This is a 45 degree rhombus on in the region (0, 0) to (2, 2)
            # It is not obvious which of the 'left' or 'top' corners to consider
            # the 'top left'. By convention, the algorithm should pick the first
            # such occurrence in the input array.
            # The input contour is on the form (TL, BL, BR, TR)
            # The ordered contour should be as (TL, TR, BR, BL)
            np.array([[0, 1], [1, 2], [2, 1], [1, 0]]),
            np.array([[0, 1], [1, 0], [2, 1], [1, 2]]),
        ],
    ],
)
def test_order_contour(contour: np.ndarray, expected_contour: np.ndarray):
    """
    Test the function to order a contour of four corners so that they will be
    in the order [Top Left, Top Right, Bottom Right, Bottom Left].
    """
    actual_contour = scanner.order_contour(contour=contour)
    np.testing.assert_array_equal(actual_contour, expected_contour)


def test_find_bright_documents_returns_separate_pages_in_reading_order():
    image = np.full((900, 1200, 3), 18, dtype=np.uint8)
    left = np.array([[80, 100], [545, 80], [560, 800], [95, 820]])
    right = np.array([[650, 85], [1120, 110], [1100, 815], [635, 790]])
    cv2.fillConvexPoly(image, left, (235, 235, 235))
    cv2.fillConvexPoly(image, right, (245, 245, 245))

    contours = scanner.find_bright_documents(image)

    assert len(contours) == 2
    centers = [np.mean(contour, axis=0) for contour in contours]
    assert centers[0][0] < centers[1][0]
    for contour in contours:
        assert contour.shape == (4, 2)


def test_crop_contour_without_perspective_keeps_axis_aligned_bounds():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    contour = np.array([[50, 20], [150, 30], [140, 80], [60, 70]])

    cropped, ordered = scanner.crop_contour_without_perspective(image, contour)

    assert ordered.shape == (4, 2)
    assert cropped.shape[:2] == (61, 101)


def test_fixed_page_contours_use_a3_bed_scale_and_are_centered():
    image = np.zeros((900, 1200, 3), dtype=np.uint8)
    a3 = scanner.fixed_page_contour(image, 420, 297)
    a4 = scanner.fixed_page_contour(image, 210, 297)
    a5 = scanner.fixed_page_contour(image, 148, 210)

    for contour in (a3, a4, a5):
        center = np.mean(contour, axis=0)
        np.testing.assert_allclose(center, [600, 450], atol=1)

    a3_width = a3[1][0] - a3[0][0]
    a4_width = a4[1][0] - a4[0][0]
    a4_height = a4[2][1] - a4[1][1]
    a5_width = a5[1][0] - a5[0][0]
    a5_height = a5[2][1] - a5[1][1]
    assert a4_width == pytest.approx(a3_width / 2, abs=1)
    assert a5_width / a5_height == pytest.approx(148 / 210, abs=0.01)
    assert a4_width / a4_height == pytest.approx(210 / 297, abs=0.01)


@pytest.mark.parametrize(
    "image_file, expected_contour",
    [
        [
            "tests/images/IMG_1842.jpg",
            [[47, 84], [938, 84], [950, 681], [47, 685]],
        ],
        [
            "tests/images/IMG_1843.jpg",
            [[306, 101], [725, 310], [453, 869], [31, 667]],
        ],
        [
            "tests/images/IMG_1844.jpg",
            [[363, 154], [712, 339], [405, 845], [26, 576]],
        ],
        [
            "tests/images/IMG_1845.jpg",
            [[370, 266], [697, 356], [567, 837], [175, 678]],
        ],
        [
            "tests/images/IMG_1846.jpg",
            [[285, 164], [842, 269], [806, 681], [136, 515]],
        ],
    ],
)
def test_scanner(image_file: str, expected_contour: list[tuple[int, int]]):
    """
    Test the algorithm's ability to accurately detect the contour corners of
    a few test images.
    """
    image = cv2.imread(image_file)
    scan_result = scanner.main(img=image)
    actual_contour = scan_result.contour
    assert actual_contour is not None, "No contour produced"
    assert actual_contour.shape == (4, 2), "Wrong shape contour"

    # The expected and actual contours should be ordered as (TL, TR, BR, BL)
    names = ("TL", "TR", "BR", "BL")
    max_distance = 30
    failed = []

    # For each pair of expected and actual corners, check the distance
    for p1, p2, name in zip(expected_contour, actual_contour, names):
        d = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        message = f"Corner {name}: Expected: {p1}, Actual: {p2}, Distance: {d}"
        print(message)
        # If the corner distance is too big, it is a failed corner
        if d > max_distance:
            failed.append(f"{message}: Distance > {max_distance}")

    # Assert that no corners failed, or print them if they did
    assert not failed, "Bad corners:\n" + "\n".join(failed)
