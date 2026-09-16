"""
face_module.py
---------------
Functional Module 1: Face Detection & Privacy Anonymisation.

Uses OpenCV's bundled Haar Cascade classifier (`haarcascade_frontalface_default.xml`,
trained by Viola-Jones on frontal face images) to locate faces in an image or
video, then Gaussian-blurs each detected region to anonymise identity.

Model selection rationale:
  Haar Cascades are lightweight, ship inside opencv-python (no external
  download / internet access required at run time), and run comfortably on
  CPU in real time -- appropriate for a CLI tool that must be reproducible
  in any evaluator's environment.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from src.image_preprocessing import equalize_histogram, gaussian_blur_region, to_grayscale
from src.logger_setup import get_logger

logger = get_logger(__name__)

Rect = Tuple[int, int, int, int]


@dataclass
class FaceDetectionResult:
    faces: List[Rect]
    annotated_image: np.ndarray


class FaceDetector:
    """Wraps a Haar Cascade face detector with sane error handling."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5,
                 blur_kernel: int = 35):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.isfile(cascade_path):
            raise RuntimeError(
                f"Haar cascade file not found at '{cascade_path}'. "
                "Your OpenCV installation may be corrupted."
            )
        self._cascade = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.blur_kernel = blur_kernel

    def detect(self, image: np.ndarray) -> List[Rect]:
        """Return a list of (x, y, w, h) bounding boxes for detected faces."""
        if image is None or image.size == 0:
            raise ValueError("FaceDetector.detect() received an empty image.")

        gray = to_grayscale(image)
        gray = equalize_histogram(gray)  # improves detection in poor lighting

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30),
        )
        return [tuple(map(int, f)) for f in faces]

    def anonymise(self, image: np.ndarray, draw_boxes: bool = True) -> FaceDetectionResult:
        """
        Detect faces and blur them for privacy. If `draw_boxes` is True,
        a thin rectangle outline is drawn around each blurred region so a
        human reviewer can audit what was redacted.
        """
        faces = self.detect(image)
        output = image.copy()

        for (x, y, w, h) in faces:
            output = gaussian_blur_region(output, x, y, w, h, kernel=self.blur_kernel)
            if draw_boxes:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        logger.info("Face module: detected and anonymised %d face(s).", len(faces))
        return FaceDetectionResult(faces=faces, annotated_image=output)


def process_image_file(input_path: str, output_path: str, detector: FaceDetector) -> int:
    """Convenience wrapper: read an image file, anonymise faces, write result."""
    from src.image_preprocessing import load_image

    image = load_image(input_path)
    result = detector.anonymise(image)
    ok = cv2.imwrite(output_path, result.annotated_image)
    if not ok:
        raise IOError(f"Failed to write output image to '{output_path}'.")
    return len(result.faces)


def process_video_file(input_path: str, output_path: str, detector: FaceDetector,
                        frame_skip: int = 1) -> int:
    """
    Anonymise faces in every `frame_skip`-th frame of a video and write the
    annotated result to `output_path`. Returns the total number of faces
    blurred across the whole video (Performance NFR: frame_skip lets a user
    trade accuracy for speed on long videos).
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

    total_faces = 0
    frame_idx = 0
    last_result_frame = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_skip == 0:
                result = detector.anonymise(frame)
                total_faces += len(result.faces)
                last_result_frame = result.annotated_image
            else:
                last_result_frame = frame

            writer.write(last_result_frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    logger.info("Face module: processed %d frames, %d total face detections.",
                frame_idx, total_faces)
    return total_faces
