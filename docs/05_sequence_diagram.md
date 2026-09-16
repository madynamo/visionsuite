# Sequence Diagram -- `face-blur` command (representative example)

```mermaid
sequenceDiagram
    actor U as User
    participant CLI as main.py (CLI)
    participant CFG as AppConfig
    participant FD as FaceDetector
    participant IP as image_preprocessing
    participant CV as OpenCV (Haar Cascade)
    participant RR as RunReport

    U->>CLI: python main.py face-blur --input photo.jpg
    CLI->>CFG: AppConfig.load("config.yaml")
    CFG-->>CLI: AppConfig instance
    CLI->>CLI: validate input path exists
    CLI->>FD: FaceDetector(scale_factor, min_neighbors, blur_kernel)
    CLI->>FD: process_image_file(input, output, detector)
    FD->>IP: load_image(path)
    IP-->>FD: BGR ndarray
    FD->>IP: to_grayscale(image)
    IP-->>FD: grayscale ndarray
    FD->>IP: equalize_histogram(gray)
    IP-->>FD: enhanced grayscale
    FD->>CV: detectMultiScale(gray)
    CV-->>FD: list of (x, y, w, h) boxes
    loop for each detected face
        FD->>IP: gaussian_blur_region(image, box, kernel)
        IP-->>FD: image with blurred region
    end
    FD-->>CLI: FaceDetectionResult(faces, annotated_image)
    CLI->>CLI: cv2.imwrite(output_path, annotated_image)
    CLI->>RR: new RunReport(command, input, output_dir)
    CLI->>RR: add_result("faces_detected", n)
    CLI->>RR: save_json(), append_csv()
    RR-->>CLI: report persisted
    CLI-->>U: "Done. Output written to: outputs/photo_face_blurred.png"
```

The `count-people` and `train-digits` / `predict-digit` flows follow the
same shape: CLI parses input → loads config → delegates to the relevant
module → module uses shared preprocessing + the underlying CV/ML library →
result is written to disk and summarised in a `RunReport`.
