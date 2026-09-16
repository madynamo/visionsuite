# Use Case Diagram

```mermaid
flowchart LR
    User((CLI User /<br/>Evaluator))

    subgraph VisionSuite["VisionSuite System"]
        UC1([Anonymise faces<br/>in image or video])
        UC2([Count pedestrians<br/>in image or video])
        UC3([Train digit<br/>classification model])
        UC4([Classify a handwritten<br/>digit image])
        UC5([View structured run<br/>report / logs])
        UC6([Configure detection<br/>parameters via YAML])
    end

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6

    UC1 -.includes.-> UC5
    UC2 -.includes.-> UC5
    UC3 -.includes.-> UC5
    UC4 -.includes.-> UC5
    UC4 -.requires.-> UC3
```

**Actors:** A single actor -- the CLI user (which may be a human developer or
an automated evaluation script) -- interacts with the system entirely
through the `main.py` command-line interface.

**Notes:**
- `Classify a handwritten digit` requires that `Train digit classification
  model` has been run at least once (a trained `.pkl` model must exist).
- Every use case implicitly includes generating a structured run report,
  supporting auditability.
