"""Unit tests for src/people_counter_module.py"""

import numpy as np
import pytest

from src.people_counter_module import PeopleCounter, _non_max_suppression


@pytest.fixture
def counter():
    return PeopleCounter()


def test_detect_rejects_empty_image(counter):
    with pytest.raises(ValueError):
        counter.detect(np.array([]))


def test_detect_returns_list_on_blank_image(counter):
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    boxes = counter.detect(blank)
    assert isinstance(boxes, list)
    assert boxes == []  # a blank frame has no pedestrians


def test_count_and_annotate_matches_box_count(counter):
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    result = counter.count_and_annotate(blank)
    assert result.count == len(result.boxes)
    assert result.annotated_image.shape == blank.shape


def test_non_max_suppression_removes_full_overlap():
    boxes = np.array([[10, 10, 50, 100], [12, 12, 50, 100]])  # near-identical boxes
    kept = _non_max_suppression(boxes, overlap_thresh=0.5)
    assert len(kept) == 1


def test_non_max_suppression_keeps_disjoint_boxes():
    boxes = np.array([[0, 0, 20, 20], [200, 200, 20, 20]])
    kept = _non_max_suppression(boxes, overlap_thresh=0.5)
    assert len(kept) == 2


def test_non_max_suppression_handles_empty_input():
    result = _non_max_suppression(np.array([]))
    assert len(result) == 0
