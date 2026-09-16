# System Architecture Diagram

VisionSuite follows a layered, modular architecture. The CLI layer never
talks to OpenCV/sklearn directly -- it always goes through a module wrapper,
which keeps the system testable and lets any module be replaced (e.g. HOG
swapped for a deep detector) without touching the CLI or the other modules.

```mermaid
flowchart TB
    subgraph CLI["CLI Layer"]
        MAIN["main.py (argparse CLI)"]
    end

    subgraph CONFIG["Configuration Layer"]
        CFG["config_loader.py"]
        YAML["config.yaml"]
    end

    subgraph MODULES["Functional Modules Layer"]
        FACE["face_module.py<br/>(Face Detection & Blur)"]
        PEOPLE["people_counter_module.py<br/>(Pedestrian Detection & Count)"]
        DIGIT["digit_classifier/<br/>(train.py, predict.py)"]
    end

    subgraph SHARED["Shared Utilities Layer"]
        PREP["image_preprocessing.py"]
        LOG["logger_setup.py"]
    end

    subgraph OUTPUT["Output & Reporting Layer"]
        REPORT["report_generator.py"]
        FILES["outputs/ (annotated media,<br/>JSON/CSV reports, logs)"]
        MODELS["models/ (trained .pkl)"]
    end

    subgraph EXT["External Libraries"]
        CV2["OpenCV (Haar Cascade,<br/>HOG+SVM, image ops)"]
        SK["scikit-learn (SVM classifier,<br/>digits dataset)"]
    end

    MAIN --> CFG --> YAML
    MAIN --> FACE
    MAIN --> PEOPLE
    MAIN --> DIGIT
    MAIN --> REPORT

    FACE --> PREP
    PEOPLE --> PREP
    FACE --> CV2
    PEOPLE --> CV2
    DIGIT --> SK

    FACE --> LOG
    PEOPLE --> LOG
    DIGIT --> LOG
    REPORT --> LOG

    REPORT --> FILES
    DIGIT --> MODELS
    FACE --> FILES
    PEOPLE --> FILES
```

**Layer responsibilities**

| Layer | Responsibility |
|---|---|
| CLI Layer | Parses arguments, dispatches to the correct module, owns the top-level error boundary |
| Configuration Layer | Loads and validates `config.yaml` into a typed `AppConfig` object |
| Functional Modules Layer | Implements the three core CV capabilities (detection, counting, classification) |
| Shared Utilities Layer | Reusable image-processing primitives and consistent logging, used by every module |
| Output & Reporting Layer | Persists annotated media, structured run reports (JSON/CSV) and rotating logs |
| External Libraries | OpenCV and scikit-learn provide the underlying CV/ML algorithms |
