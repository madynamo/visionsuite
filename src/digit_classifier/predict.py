"""
predict.py
----------
Runs inference with the trained digit classifier on a new image.

The input is expected to be a roughly-square grayscale image of a single
handwritten digit on a plain background (white digit on dark background,
or vice versa -- both are auto-detected and normalised). It is resized to
8x8 to match the training data's resolution before classification.
"""

import argparse
import os

import cv2
import numpy as np

from src.digit_classifier.model_utils import load_model, preprocess_for_inference
from src.logger_setup import get_logger

logger = get_logger(__name__)


def _normalise_digit_image(gray: np.ndarray) -> np.ndarray:
    """
    Ensure the digit is drawn as bright strokes on a dark background
    (the convention used by sklearn's digits dataset) and resize to 8x8.
    """
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)

    # Heuristic: if the image is mostly bright (light background), invert it
    # so the digit stroke is bright and background is dark, matching training data.
    if np.mean(resized) > 127:
        resized = 255 - resized

    return resized


def predict_digit(image_path: str, model_path: str = "models/digit_classifier.pkl"):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: '{image_path}'.")

    trained = load_model(model_path)

    raw = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Could not read image '{image_path}' as grayscale.")

    normalised = _normalise_digit_image(raw)
    features = preprocess_for_inference(normalised)

    prediction = trained.model.predict(features)[0]
    probabilities = trained.model.predict_proba(features)[0]
    confidence = float(np.max(probabilities))

    logger.info("Predicted digit %d (confidence %.2f%%) for '%s'.",
                prediction, confidence * 100, image_path)

    return {
        "predicted_digit": int(prediction),
        "confidence": confidence,
        "all_probabilities": probabilities.tolist(),
        "model_accuracy_on_test_set": trained.metadata.get("accuracy"),
    }


def main():
    parser = argparse.ArgumentParser(description="Classify a handwritten digit image.")
    parser.add_argument("image_path", type=str, help="Path to a digit image (PNG/JPG).")
    parser.add_argument("--model", type=str, default="models/digit_classifier.pkl")
    args = parser.parse_args()

    result = predict_digit(args.image_path, args.model)
    print(f"Predicted digit : {result['predicted_digit']}")
    print(f"Confidence      : {result['confidence'] * 100:.2f}%")
    print(f"Model test acc. : {result['model_accuracy_on_test_set']:.4f}")


if __name__ == "__main__":
    main()
