"""Unit tests for src/image_preprocessing.py"""

import numpy as np
import pytest

from src.image_preprocessing import (detect_edges, equalize_histogram,
                                      gaussian_blur_region, resize_max_dimension,
                                      to_grayscale)


@pytest.fixture
def color_image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(100, 150, 3), dtype=np.uint8)


def test_to_grayscale_converts_color_image(color_image):
    gray = to_grayscale(color_image)
    assert gray.shape == (100, 150)


def test_to_grayscale_passthrough_for_already_gray(color_image):
    gray = to_grayscale(color_image)
    gray_again = to_grayscale(gray)
    assert np.array_equal(gray, gray_again)


def test_to_grayscale_rejects_empty_image():
    with pytest.raises(ValueError):
        to_grayscale(np.array([]))


def test_equalize_histogram_requires_grayscale(color_image):
    with pytest.raises(ValueError):
        equalize_histogram(color_image)


def test_equalize_histogram_returns_same_shape(color_image):
    gray = to_grayscale(color_image)
    equalized = equalize_histogram(gray)
    assert equalized.shape == gray.shape


def test_detect_edges_returns_binary_like_output(color_image):
    gray = to_grayscale(color_image)
    edges = detect_edges(gray)
    assert edges.shape == gray.shape
    assert set(np.unique(edges)).issubset({0, 255})


def test_resize_max_dimension_shrinks_large_image():
    big = np.zeros((2000, 1000, 3), dtype=np.uint8)
    resized = resize_max_dimension(big, max_dim=500)
    assert max(resized.shape[:2]) == 500


def test_resize_max_dimension_leaves_small_image_untouched():
    small = np.zeros((100, 50, 3), dtype=np.uint8)
    resized = resize_max_dimension(small, max_dim=500)
    assert resized.shape == small.shape


def test_gaussian_blur_region_modifies_only_target_area():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:] = 100
    blurred = gaussian_blur_region(image.copy(), x=10, y=10, w=20, h=20, kernel=15)
    # Region outside the blur box should be untouched.
    assert np.array_equal(blurred[0:5, 0:5], image[0:5, 0:5])
