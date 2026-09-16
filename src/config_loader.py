"""
config_loader.py
-----------------
Loads and validates the project's YAML configuration file.

Centralising configuration (instead of hard-coding thresholds/paths inside
each module) supports the Maintainability and Usability non-functional
requirements: a user can change detection sensitivity, output folders, etc.
without touching any source code.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

from src.logger_setup import get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


class ConfigError(Exception):
    """Raised when the configuration file is missing or malformed."""


@dataclass
class AppConfig:
    """Typed, validated view over the raw YAML configuration dictionary."""

    face_scale_factor: float = 1.1
    face_min_neighbors: int = 5
    face_blur_kernel: int = 35

    people_hit_threshold: float = 0.0
    people_win_stride: int = 8

    digit_model_path: str = "models/digit_classifier.pkl"
    digit_test_size: float = 0.2
    digit_random_state: int = 42

    output_dir: str = "outputs"
    log_level: str = "INFO"

    raw: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def load(path: str = DEFAULT_CONFIG_PATH) -> "AppConfig":
        if not os.path.isfile(path):
            raise ConfigError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse YAML config '{path}': {exc}") from exc

        try:
            face = raw.get("face_detection", {})
            people = raw.get("people_detection", {})
            digit = raw.get("digit_classifier", {})
            general = raw.get("general", {})

            cfg = AppConfig(
                face_scale_factor=float(face.get("scale_factor", 1.1)),
                face_min_neighbors=int(face.get("min_neighbors", 5)),
                face_blur_kernel=int(face.get("blur_kernel", 35)),
                people_hit_threshold=float(people.get("hit_threshold", 0.0)),
                people_win_stride=int(people.get("win_stride", 8)),
                digit_model_path=str(digit.get("model_path", "models/digit_classifier.pkl")),
                digit_test_size=float(digit.get("test_size", 0.2)),
                digit_random_state=int(digit.get("random_state", 42)),
                output_dir=str(general.get("output_dir", "outputs")),
                log_level=str(general.get("log_level", "INFO")),
                raw=raw,
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid value in configuration file: {exc}") from exc

        logger.debug("Configuration loaded from %s", path)
        return cfg
