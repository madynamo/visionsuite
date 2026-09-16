"""
image_preprocessing.py
-----------------------
Shared, reusable image-preprocessing utilities used by every detection
module in VisionSuite. Keeping these functions in one place (rather than
duplicating them inside the face/people/digit modules) is the project's
main Maintainability and DRY-design decision.

Functions here never crash on bad input silently -- they raise a
ValueError with a clear message, which callers can catch and log
(Non-Functional Requirement: Error Handling Strategy).
"""

from typing import Tuple

import cv2
import numpy as np

from src.logger_setup import get_logger

logger = get_logger(__name__)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR numpy array, or raise ValueError."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Could not read image file: '{path}'. "
                          f"Check the path and that it is a valid image format.")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale."""
    if image is None or image.size == 0:
        raise ValueError("to_grayscale() received an empty image.")
    if len(image.shape) == 2:
        return image  # already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray, strength: int = 7) -> np.ndarray:
    """Apply fast non-local-means denoising to reduce sensor/compression noise."""
    if len(image.shape) == 2:
        return cv2.fastNlMeansDenoising(image, h=strength)
    return cv2.fastNlMeansDenoisingColored(image, h=strength, hColor=strength)


def equalize_histogram(gray_image: np.ndarray) -> np.ndarray:
    """Improve contrast using CLAHE (adaptive histogram equalisation)."""
    if len(gray_image.shape) != 2:
        raise ValueError("equalize_histogram() expects a single-channel image.")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)


def detect_edges(gray_image: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Run Canny edge detection on a grayscale image."""
    if len(gray_image.shape) != 2:
        raise ValueError("detect_edges() expects a single-channel image.")
    return cv2.Canny(gray_image, low, high)


def resize_max_dimension(image: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """
    Downscale an image so its largest side is at most `max_dim` pixels,
    preserving aspect ratio. Keeps processing fast on very large inputs
    (Non-Functional Requirement: Performance / Resource Efficiency).
    """
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= max_dim:
        return image
    scale = max_dim / float(largest)
    new_size: Tuple[int, int] = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def gaussian_blur_region(image: np.ndarray, x: int, y: int, w: int, h: int,
                          kernel: int = 35) -> np.ndarray:
    """
    Blur a rectangular region of `image` in place and return it.
    Used for privacy-preserving face anonymisation.
    """
    kernel = kernel if kernel % 2 == 1 else kernel + 1  # kernel size must be odd
    x, y = max(0, x), max(0, y)
    roi = image[y:y + h, x:x + w]
    if roi.size == 0:
        return image
    blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)
    image[y:y + h, x:x + w] = blurred
    return image
