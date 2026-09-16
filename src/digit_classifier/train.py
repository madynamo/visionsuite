"""
train.py
--------
Trains and evaluates the handwritten-digit SVM classifier.

Evaluation methodology:
  The 1,797-sample dataset is split into train/test partitions (default
  80/20, stratified by class) using a fixed random seed for reproducibility.
  We report overall accuracy, a full per-class precision/recall/F1
  classification report, and a confusion matrix -- saved both as a text
  report and as a PNG heatmap for inclusion in the project report.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless CLI environment, no display server
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.digit_classifier.model_utils import TrainedModel, save_model
from src.logger_setup import get_logger

logger = get_logger(__name__)


def train_and_evaluate(test_size: float = 0.2, random_state: int = 42,
                        model_out_path: str = "models/digit_classifier.pkl",
                        report_dir: str = "outputs") -> dict:
    logger.info("Loading sklearn 'load_digits' dataset (1,797 samples, 10 classes)...")
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Train/test split: %d train samples, %d test samples.",
                len(X_train), len(X_test))

    clf = SVC(kernel="rbf", gamma=0.001, C=10, probability=True,
              random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("Test accuracy: %.4f", accuracy)

    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "digit_classifier_evaluation.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("VisionSuite - Digit Classifier Evaluation\n")
        fh.write("=" * 45 + "\n\n")
        fh.write(f"Dataset: sklearn.datasets.load_digits (n_samples={len(X)})\n")
        fh.write(f"Train/test split: {1 - test_size:.0%}/{test_size:.0%}, "
                  f"random_state={random_state}\n\n")
        fh.write(f"Overall accuracy: {accuracy:.4f}\n\n")
        fh.write("Per-class precision / recall / F1:\n")
        fh.write(report_text)
        fh.write("\nConfusion matrix (rows=true label, cols=predicted):\n")
        fh.write(np.array2string(cm))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Digit Classifier Confusion Matrix (acc={accuracy:.3f})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = os.path.join(report_dir, "digit_classifier_confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    trained = TrainedModel(
        model=clf,
        metadata={
            "accuracy": accuracy,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_size": test_size,
            "random_state": random_state,
            "classes": list(range(10)),
        },
    )
    save_model(trained, model_out_path)

    logger.info("Evaluation report written to '%s'.", report_path)
    logger.info("Confusion matrix image written to '%s'.", cm_path)

    return {"accuracy": accuracy, "report_path": report_path, "cm_path": cm_path,
            "model_path": model_out_path}


def main():
    parser = argparse.ArgumentParser(description="Train the VisionSuite digit classifier.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-out", type=str, default="models/digit_classifier.pkl")
    parser.add_argument("--report-dir", type=str, default="outputs")
    args = parser.parse_args()

    result = train_and_evaluate(
        test_size=args.test_size,
        random_state=args.random_state,
        model_out_path=args.model_out,
        report_dir=args.report_dir,
    )
    print(f"Training complete. Accuracy: {result['accuracy']:.4f}")
    print(f"Model saved to: {result['model_path']}")
    print(f"Evaluation report: {result['report_path']}")
    print(f"Confusion matrix image: {result['cm_path']}")


if __name__ == "__main__":
    main()
