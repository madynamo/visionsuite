"""
generate_sample_media.py
-------------------------
Generates small, synthetic sample images/video into sample_data/ so that an
evaluator (or a fresh clone of this repo) can immediately try every
VisionSuite module end-to-end without needing to source their own
photos/videos or a licensed dataset.

Note: synthetic cartoon shapes will NOT be detected by the real-world
face/people detectors (they are trained on real photographs) -- that is
expected. For a meaningful face/people-detection demo, point the CLI at a
real photo/video of your own using --input. The synthetic digit images
generated here, however, are directly usable with the digit classifier
because they follow the same 8x8 stroke convention as the training data.
"""

import os

import cv2
import numpy as np
from sklearn.datasets import load_digits

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_placeholder_scene(path: str) -> None:
    """A simple synthetic 'scene' image, useful for smoke-testing the
    face/people pipelines without crashing (won't yield real detections)."""
    img = np.full((480, 640, 3), 30, dtype=np.uint8)
    cv2.putText(img, "VisionSuite sample scene", (60, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    cv2.putText(img, "(use a real photo for actual face/people detection)",
                (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
    cv2.imwrite(path, img)


def generate_sample_digits(out_dir: str, n: int = 5) -> None:
    """Export a handful of real digit samples from sklearn's dataset as PNGs
    (upscaled for visibility) that can be fed straight into predict.py."""
    digits = load_digits()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(digits.images), size=n, replace=False)
    for i, idx in enumerate(indices):
        image_8x8 = digits.images[idx].astype(np.uint8)
        # Scale 0-16 pixel range up to 0-255 for a viewable PNG.
        image_8x8_255 = (image_8x8 * (255 / 16)).astype(np.uint8)
        upscaled = cv2.resize(image_8x8_255, (128, 128), interpolation=cv2.INTER_NEAREST)
        label = int(digits.target[idx])
        out_path = os.path.join(out_dir, f"sample_digit_{i}_label_{label}.png")
        cv2.imwrite(out_path, upscaled)
        print(f"Wrote {out_path} (true label = {label})")


def main():
    scene_path = os.path.join(HERE, "sample_scene.png")
    digits_dir = os.path.join(HERE, "sample_digits")

    generate_placeholder_scene(scene_path)
    print(f"Wrote {scene_path}")

    generate_sample_digits(digits_dir, n=5)
    print("\nSample media generated successfully.")


if __name__ == "__main__":
    main()
