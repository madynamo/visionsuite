# VisionSuite

**A modular, offline, command-line computer-vision analysis toolkit** --
face anonymisation, pedestrian counting, and handwritten-digit
classification, all from a single CLI, with structured logging and
audit-friendly run reports.

> Built as a Computer Vision course project. See [`statement.md`](statement.md)
> for the full problem statement, scope, and target users.

---

## Overview

VisionSuite provides three independent but consistently-designed computer
vision capabilities, reachable through one command-line entry point
(`main.py`):

| Module | What it does | Technique used |
|---|---|---|
| **Face Blur** | Detects faces in an image/video and blurs them for privacy | Haar Cascade classifier (OpenCV) |
| **People Counter** | Detects and counts pedestrians in an image/video | HOG descriptor + linear SVM (OpenCV) |
| **Digit Classifier** | Trains, evaluates, and runs inference for handwritten digit recognition | RBF-kernel SVM (scikit-learn) |

All three models are bundled with their respective libraries (OpenCV /
scikit-learn) -- **no internet access or external dataset download is
required to run this project**, which keeps it fully reproducible for
evaluation.

## Features

- Detect & blur faces in images **and** videos, with adjustable blur
  strength and frame-skip for speed on long videos.
- Detect & count pedestrians in images **and** videos, with non-maximum
  suppression to avoid double-counting overlapping detections.
- Train a handwritten-digit classifier from scratch and see its accuracy,
  per-class precision/recall/F1 report, and a confusion-matrix plot.
- Classify a new handwritten digit image using the trained model.
- Centralised YAML configuration (`config.yaml`) -- tune detection
  sensitivity without touching any code.
- Structured, timestamped run reports (`outputs/last_run_report.json`,
  `outputs/run_history.csv`) and rotating application logs
  (`outputs/logs/visionsuite.log`) after every command.
- Graceful error handling everywhere: missing files, bad config, or a
  missing trained model all produce a clear message and a non-zero exit
  code instead of a stack trace.
- 24 automated unit tests (`pytest`) covering every module's core logic.
- Synthetic sample data generator so you can try every command immediately
  after cloning, with no external images/videos required.

## Technologies / Tools Used

- **Python 3.10+**
- **OpenCV** (`opencv-python`) -- image/video I/O, Haar Cascade face
  detection, HOG pedestrian detection, all image-processing primitives
- **scikit-learn** -- SVM classifier, the `digits` dataset, train/test
  split, evaluation metrics
- **NumPy** -- array/image manipulation
- **Matplotlib** -- confusion-matrix visualisation
- **PyYAML** -- configuration file parsing
- **pytest** -- automated unit testing
- Standard library: `argparse`, `logging` (with rotation), `dataclasses`,
  `json`, `csv`, `pickle`

## Project Structure

```
visionsuite/
├── main.py                          # CLI entry point (all subcommands)
├── config.yaml                      # tunable runtime configuration
├── requirements.txt
├── statement.md                     # problem statement / scope / users
├── README.md                        # this file
├── src/
│   ├── logger_setup.py              # centralised, rotating logging
│   ├── config_loader.py             # loads + validates config.yaml
│   ├── image_preprocessing.py       # shared CV utility functions
│   ├── face_module.py               # Module 1: face detection & blur
│   ├── people_counter_module.py     # Module 2: pedestrian detection & count
│   ├── report_generator.py          # JSON/CSV run-report writer
│   └── digit_classifier/
│       ├── model_utils.py           # save/load trained model, preprocessing
│       ├── train.py                 # Module 3: train + evaluate SVM
│       └── predict.py               # Module 3: run inference
├── sample_data/
│   └── generate_sample_media.py     # generates demo images (no downloads needed)
├── tests/                           # pytest unit tests (24 tests)
├── docs/                            # architecture & UML diagrams (Mermaid)
├── models/                          # trained model artefacts saved here
└── outputs/                         # annotated media, reports, logs saved here
```

## Setup & Installation

### 1. Prerequisites

- Python 3.10 or newer
- `pip` (Python package installer)
- No GPU, internet access, or external dataset download is required.

### 2. Clone the repository

```bash
git clone https://github.com/madynamo/visionsuite.git
cd visionsuite
```

### 3. (Recommended) create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Generate sample media (optional, but recommended for a first run)

```bash
python3 sample_data/generate_sample_media.py
```

This creates `sample_data/sample_scene.png` and five real handwritten-digit
sample images under `sample_data/sample_digits/`.

## How to Run

All commands are run through `main.py`. Use `--help` at any level for
details:

```bash
python3 main.py --help
python3 main.py face-blur --help
```

### Module 1 -- Face Detection & Blur

```bash
# On an image
python3 main.py face-blur --input path/to/photo.jpg

# On a video, processing every 2nd frame for speed
python3 main.py face-blur --input path/to/video.mp4 --frame-skip 2

# Custom output path
python3 main.py face-blur --input photo.jpg --output outputs/redacted.png
```

> Note: the synthetic `sample_scene.png` generated above is a smoke-test
> image with no real face in it (by design, to avoid bundling photos of
> real people in the repository). To see real detections, point `--input`
> at any real photo or video you have locally, e.g. one containing people's
> faces.

### Module 2 -- Pedestrian Detection & Counting

```bash
# On an image
python3 main.py count-people --input path/to/street_photo.jpg

# On a video
python3 main.py count-people --input path/to/video.mp4 --frame-skip 3
```

### Module 3 -- Handwritten Digit Classification

```bash
# Step 1: train and evaluate the model (creates models/digit_classifier.pkl)
python3 main.py train-digits

# Step 2: classify a sample digit image
python3 main.py predict-digit --input sample_data/sample_digits/sample_digit_0_label_0.png
```

Expected output after training (numbers may vary slightly by machine):

```
Training complete. Test accuracy: 0.9917
Model saved to: models/digit_classifier.pkl
```

Training also writes:
- `outputs/digit_classifier_evaluation.txt` -- full precision/recall/F1
  report + confusion matrix (text)
- `outputs/digit_classifier_confusion_matrix.png` -- confusion matrix plot

### Every run produces a report

After any command, check:

```bash
cat outputs/last_run_report.json     # structured summary of the last run
cat outputs/run_history.csv          # append-only history of all runs
tail outputs/logs/visionsuite.log    # detailed application log
```

### Using a custom configuration

```bash
python3 main.py --config path/to/my_config.yaml count-people --input photo.jpg
```

See [`config.yaml`](config.yaml) for all tunable parameters (detection
thresholds, blur strength, output directory, log level, etc.).

## Testing

The project ships with 24 automated unit tests covering input validation,
error handling, and core algorithm behaviour (non-max suppression,
preprocessing correctness, model save/load, and an end-to-end training
smoke test).

```bash
# Run the full test suite
python3 -m pytest -v

# Run only the fast tests (skip the full model-training test)
python3 -m pytest -v -m "not slow"
```

Expected result: all tests pass (`24 passed`).

## Design & Documentation

Detailed design artefacts -- system architecture, workflow, use-case,
class, sequence, and storage-schema diagrams (in Mermaid format, rendered
natively by GitHub) -- are in the [`docs/`](docs) folder:

1. [`docs/01_architecture_diagram.md`](docs/01_architecture_diagram.md)
2. [`docs/02_workflow_diagram.md`](docs/02_workflow_diagram.md)
3. [`docs/03_use_case_diagram.md`](docs/03_use_case_diagram.md)
4. [`docs/04_class_diagram.md`](docs/04_class_diagram.md)
5. [`docs/05_sequence_diagram.md`](docs/05_sequence_diagram.md)
6. [`docs/06_storage_schema_design.md`](docs/06_storage_schema_design.md)

## Non-Functional Requirements

| Requirement | How it's addressed |
|---|---|
| **Performance** | `resize_max_dimension()` caps processing resolution; `--frame-skip` trades accuracy for speed on video |
| **Reliability** | Every CLI command validates inputs up front and is wrapped in a top-level error boundary in `main.py` so failures never produce a raw traceback |
| **Usability** | Single consistent CLI (`argparse` with per-command `--help`); YAML config instead of hard-coded values |
| **Maintainability** | Modular package layout; shared `image_preprocessing.py` avoids duplication; type-hinted dataclasses for results |
| **Error Handling** | Explicit `ValueError` / `FileNotFoundError` / `ConfigError` exceptions with actionable messages; all caught and logged, never silently swallowed |
| **Logging & Monitoring** | Rotating file handler + console handler via `logger_setup.py`; every run also gets a structured JSON/CSV report |
| **Resource Efficiency** | Log rotation caps disk usage; image resizing avoids unnecessary memory/CPU use on very large inputs |
| **Scalability** | Modules are independent and stateless per call, so they can be invoked repeatedly in a batch/pipeline script over many files |

## Dataset Description (Digit Classifier)

The digit classifier uses scikit-learn's bundled `load_digits` dataset:
1,797 samples of 8x8 grayscale images of handwritten digits (0-9), derived
from the UCI ML "Optical Recognition of Handwritten Digits" dataset. It
ships with `scikit-learn` (no download needed), keeping the project fully
reproducible offline. See [`src/digit_classifier/model_utils.py`](src/digit_classifier/model_utils.py)
and [`train.py`](src/digit_classifier/train.py) docstrings for the full
model-selection rationale and evaluation methodology.

## Known Limitations

- Haar Cascade and HOG+SVM are classical detectors -- they are fast and
  dependency-light but less accurate than modern deep-learning detectors,
  especially on non-frontal faces or unusual poses/lighting.
- The digit classifier expects a single, roughly-centred digit per image
  (matching the training data's convention); it does not perform
  multi-digit segmentation from a photo of a full page.
- Video processing runs on CPU; very long/high-resolution videos will take
  a proportionally long time (mitigated by `--frame-skip`).

## License

This project was created for academic coursework submission.
