# Storage / Schema Design

VisionSuite is a stateless CLI tool and does not use a relational database
-- there are no multi-user records or foreign-key relationships to model,
so a traditional ER diagram is not applicable. Persistent state is limited
to three flat-file stores, whose schemas are defined below.

## 1. Run Report (`outputs/last_run_report.json`)

One JSON document per invocation, overwritten each run:

```json
{
  "command": "face-blur",
  "input_path": "sample_data/sample_scene.png",
  "output_dir": "outputs",
  "timestamp": "2026-09-16T09:45:51",
  "results": {
    "faces_detected": 0,
    "output_path": "outputs/sample_scene_face_blurred.png",
    "elapsed_seconds": 0.312
  },
  "errors": [],
  "status": "SUCCESS"
}
```

| Field | Type | Description |
|---|---|---|
| `command` | string | Which CLI subcommand was run |
| `input_path` | string | Path to the input file processed |
| `output_dir` | string | Configured output directory |
| `timestamp` | ISO-8601 string | When the run started |
| `results` | object | Command-specific key/value outputs (counts, paths, timings) |
| `errors` | array of strings | Any error messages captured during the run |
| `status` | string | `SUCCESS` or `COMPLETED_WITH_ERRORS` |

## 2. Run History Log (`outputs/run_history.csv`)

Every invocation appends one row, giving an auditable history across runs:

| Column | Type | Description |
|---|---|---|
| `timestamp` | ISO-8601 string | Run start time |
| `command` | string | CLI subcommand |
| `input_path` | string | Input file processed |
| `status` | string | `SUCCESS` / `COMPLETED_WITH_ERRORS` |
| `results_summary` | JSON string | Serialised copy of that run's `results` dict |

## 3. Trained Model Artefact (`models/digit_classifier.pkl`)

A pickled `TrainedModel` dataclass (`src/digit_classifier/model_utils.py`):

| Field | Type | Description |
|---|---|---|
| `model` | `sklearn.svm.SVC` | The fitted classifier |
| `metadata.accuracy` | float | Test-set accuracy at training time |
| `metadata.n_train` / `n_test` | int | Split sizes used |
| `metadata.test_size` | float | Train/test split ratio |
| `metadata.random_state` | int | Seed used, for reproducibility |
| `metadata.classes` | list[int] | The 10 digit classes (0-9) |

## 4. Application Log (`outputs/logs/visionsuite.log`)

A rotating text log (2 MB per file, 3 backups kept) with one line per
log record: `timestamp | level | logger name | message`, used for
monitoring and debugging (see `src/logger_setup.py`).
