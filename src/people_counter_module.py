"""
people_counter_module.py
-------------------------
Functional Module 2: Pedestrian Detection & Counting.

Uses OpenCV's built-in HOG (Histogram of Oriented Gradients) descriptor with
the pre-trained default people SVM detector. Like the face module, this
model ships inside opencv-python, so the tool works fully offline and is
reproducible on any evaluator's machine.

Model selection rationale:
  HOG+SVM is a classical, well-understood computer-vision pipeline (feature
  engineering + a linear classifier) that is appropriate for a coursework
  project that must demonstrate understanding of core CV concepts, rather
  than treating detection as an opaque black box. It is CPU-friendly and
  needs no external weight download.
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from src.image_preprocessing import resize_max_dimension
from src.logger_setup import get_logger

logger = get_logger(__name__)

Rect = Tuple[int, int, int, int]


def _non_max_suppression(boxes: np.ndarray, overlap_thresh: float = 0.65) -> np.ndarray:
    """Simple greedy NMS to collapse overlapping duplicate detections."""
    if len(boxes) == 0:
        return np.array([])

    boxes = boxes.astype(float)
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(y2)

    keep = []
    while len(order) > 0:
        i = order[-1]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[order[:-1]]
        order = order[:-1][overlap < overlap_thresh]

    return boxes[keep].astype(int)


@dataclass
class PeopleDetectionResult:
    boxes: List[Rect]
    count: int
    annotated_image: np.ndarray


class PeopleCounter:
    """Wraps OpenCV's HOG people detector with resizing + NMS + annotation."""

    def __init__(self, hit_threshold: float = 0.0, win_stride: int = 8):
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.hit_threshold = hit_threshold
        self.win_stride = win_stride

    def detect(self, image: np.ndarray) -> List[Rect]:
        if image is None or image.size == 0:
            raise ValueError("PeopleCounter.detect() received an empty image.")

        resized = resize_max_dimension(image, max_dim=800)
        scale_x = image.shape[1] / resized.shape[1]
        scale_y = image.shape[0] / resized.shape[0]

        boxes, _weights = self._hog.detectMultiScale(
            resized,
            winStride=(self.win_stride, self.win_stride),
            padding=(8, 8),
            scale=1.05,
            hitThreshold=self.hit_threshold,
        )

        boxes = _non_max_suppression(np.array(boxes)) if len(boxes) else np.array([])

        # Map boxes back to the original (un-resized) image coordinates.
        rescaled: List[Rect] = []
        for (x, y, w, h) in boxes:
            rescaled.append((
                int(x * scale_x), int(y * scale_y),
                int(w * scale_x), int(h * scale_y),
            ))
        return rescaled

    def count_and_annotate(self, image: np.ndarray) -> PeopleDetectionResult:
        boxes = self.detect(image)
        output = image.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(output, f"People detected: {len(boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        logger.info("People module: detected %d person(es).", len(boxes))
        return PeopleDetectionResult(boxes=boxes, count=len(boxes), annotated_image=output)


def process_image_file(input_path: str, output_path: str, counter: PeopleCounter) -> int:
    from src.image_preprocessing import load_image

    image = load_image(input_path)
    result = counter.count_and_annotate(image)
    ok = cv2.imwrite(output_path, result.annotated_image)
    if not ok:
        raise IOError(f"Failed to write output image to '{output_path}'.")
    return result.count


def process_video_file(input_path: str, output_path: str, counter: PeopleCounter,
                        frame_skip: int = 2) -> Tuple[int, float]:
    """
    Count people across a video. Returns (max_count_in_any_frame, avg_count).
    frame_skip trades speed for accuracy on long videos.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: '{input_path}'.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Could not open video writer for '{output_path}'.")

    counts = []
    frame_idx = 0
    last_frame = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_skip == 0:
                result = counter.count_and_annotate(frame)
                counts.append(result.count)
                last_frame = result.annotated_image
            else:
                last_frame = frame

            writer.write(last_frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    max_count = max(counts) if counts else 0
    avg_count = (sum(counts) / len(counts)) if counts else 0.0
    logger.info("People module: processed %d frames, max=%d avg=%.2f.",
                frame_idx, max_count, avg_count)
    return max_count, avg_count
