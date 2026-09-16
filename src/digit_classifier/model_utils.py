"""
model_utils.py
---------------
Shared helpers for the handwritten-digit classification module
(Functional Module 3 -- "Prediction/Classification").

Dataset description:
  We use scikit-learn's bundled `load_digits` dataset: 1,797 samples of
  8x8 grayscale images of handwritten digits (0-9), derived from the UCI
  ML "Optical Recognition of Handwritten Digits" dataset. It ships with
  scikit-learn (no internet download required), which keeps the project
  fully reproducible for an evaluator with no network access.

Model selection rationale:
  A Support Vector Machine with an RBF kernel is a strong classical
  baseline for small, low-resolution image datasets like this one -- it
  handles the ~64-dimensional pixel-intensity feature vectors well without
  needing the large training data a CNN would require, and trains in
  under a second on CPU, which matters for a CLI tool that must remain
  fast and dependency-light (Non-Functional Requirement: Performance).
"""

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.svm import SVC

from src.logger_setup import get_logger

logger = get_logger(__name__)


@dataclass
class TrainedModel:
    model: SVC
    metadata: Dict[str, Any]


def save_model(trained: TrainedModel, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(trained, fh)
    logger.info("Saved trained digit classifier to '%s'.", path)


def load_model(path: str) -> TrainedModel:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No trained model found at '{path}'. "
            "Run `python main.py train-digits` first."
        )
    with open(path, "rb") as fh:
        trained = pickle.load(fh)
    if not isinstance(trained, TrainedModel):
        raise ValueError(f"File '{path}' does not contain a valid TrainedModel.")
    return trained


def preprocess_for_inference(image_8x8: np.ndarray) -> np.ndarray:
    """
    Flatten and scale a single 8x8 grayscale digit image the same way the
    training data is scaled (pixel range 0-16, matching sklearn's digits
    dataset convention) before feeding it to the classifier.
    """
    if image_8x8.shape != (8, 8):
        raise ValueError(f"Expected an 8x8 grayscale patch, got shape {image_8x8.shape}.")
    scaled = (image_8x8.astype(np.float64) / 255.0) * 16.0
    return scaled.flatten().reshape(1, -1)
