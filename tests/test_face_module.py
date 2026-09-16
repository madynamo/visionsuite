"""Unit tests for src/face_module.py"""

import numpy as np
import pytest

from src.face_module import FaceDetector


@pytest.fixture
def detector():
    return FaceDetector()


def test_detector_initialises_cascade(detector):
    assert detector._cascade is not None
    assert not detector._cascade.empty()


def test_detect_rejects_empty_image(detector):
    with pytest.raises(ValueError):
        detector.detect(np.array([]))


def test_detect_returns_list_of_tuples_on_blank_image(detector):
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    faces = detector.detect(blank)
    assert isinstance(faces, list)
    # A featureless blank image should not trigger any false detections.
    assert faces == []


def test_anonymise_returns_same_shape_image(detector):
    blank = np.full((200, 200, 3), 128, dtype=np.uint8)
    result = detector.anonymise(blank)
    assert result.annotated_image.shape == blank.shape
    assert isinstance(result.faces, list)
