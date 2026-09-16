"""Unit tests for src/digit_classifier/*"""

import os

import numpy as np
import pytest

from src.digit_classifier.model_utils import (TrainedModel, load_model,
                                                preprocess_for_inference, save_model)
from src.digit_classifier.train import train_and_evaluate


def test_preprocess_for_inference_rejects_wrong_shape():
    with pytest.raises(ValueError):
        preprocess_for_inference(np.zeros((10, 10)))


def test_preprocess_for_inference_scales_and_flattens():
    patch = np.full((8, 8), 255, dtype=np.uint8)
    features = preprocess_for_inference(patch)
    assert features.shape == (1, 64)
    assert np.isclose(features.max(), 16.0)


def test_save_and_load_model_roundtrip(tmp_path):
    from sklearn.svm import SVC

    dummy = SVC()
    trained = TrainedModel(model=dummy, metadata={"accuracy": 0.99})
    path = os.path.join(tmp_path, "model.pkl")

    save_model(trained, path)
    loaded = load_model(path)

    assert loaded.metadata["accuracy"] == 0.99
    assert isinstance(loaded.model, SVC)


def test_load_model_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(os.path.join(tmp_path, "does_not_exist.pkl"))


@pytest.mark.slow
def test_train_and_evaluate_reaches_reasonable_accuracy(tmp_path):
    """
    End-to-end training smoke test: on the sklearn digits dataset, an
    RBF-kernel SVM should comfortably exceed 90% test accuracy. This test
    also exercises file-writing (model, evaluation report, confusion matrix
    image) to guard against silent I/O regressions.
    """
    model_path = os.path.join(tmp_path, "model.pkl")
    result = train_and_evaluate(test_size=0.2, random_state=42,
                                 model_out_path=model_path, report_dir=str(tmp_path))

    assert result["accuracy"] > 0.90
    assert os.path.isfile(result["model_path"])
    assert os.path.isfile(result["report_path"])
    assert os.path.isfile(result["cm_path"])
