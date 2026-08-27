---
name: agentic-ecology-camera-traps
description: >-
  Provides capabilities to run SpeciesNet detector and classifier on camera trap
  images, extract crop-level feature embeddings, and populate a Hoplite vector
  database for downstream search and agile modeling.
---

# Camera Traps Skill

Use this skill when you need to process a collection of camera trap images, run
species classification, extract vector representation embeddings, and store them
inside a Hoplite vector database.

## Workflow Overview

Follow these sequential steps:

1.  **Identify Dataset and Limits**:
    *   Locate the target camera trap images directory.
    *   Assess if a GPU is available on the system. If running on CPU-only,
        discuss with the user or apply a processing limit (e.g., first 1000
        images) to prevent the ingestion pipeline from running excessively long.
2.  **Initialize Hoplite Database**:
    *   Create a Hoplite database (`SQLiteUSearchDB`) at the destination folder.
    *   Configure it with an embedding dimension of `1280` (EfficientNet-V2 M
        feature size), the metric set to `Cos`, and the data type set to
        `float16`.
3.  **Run Ingestion Pipeline**:
    *   Instantiate the `SpeciesNetDetector` and `SpeciesNetClassifier` models.
    *   Register a PyTorch forward hook on the classifier's average pooling
        layer (`SpeciesNet/efficientnetv2-m/avg_pool/Mean_Squeeze__3825`) to
        intercept raw embeddings.
    *   For each image:
        *   Insert it into the database as a recording.
        *   Run the detector model to get bounding box coords for animal
            detections.
        *   Crop the PIL image to the bounding box, preprocess it, and run the
            classifier to extract the 1280-dim embedding vector.
        *   Cast the vector to `float16` and insert it into the database as a
            window.
4.  **Agile Modeling and Search**:
    *   Once populated, use the Hoplite database to perform vector searches
        (ranking by similarity) or train active learning classifiers on top of
        the embeddings.
5.  **Camera Trap Visualization Guidelines (M3 UI)**:
    *   **Context Preservation**: Avoid displaying raw cropped images in the
        result cards. Instead, display the original (uncropped) image inside the
        card container and draw the animal detection as a red border box overlay
        dynamically using CSS absolute positioning and percentages (e.g., `left:
        xmin * 100%`, `top: ymin * 100%`, etc.).
    *   **Full-Resolution Modal Preview**: Implement a click handler on the card
        media that triggers a floating fullscreen modal containing the uncropped
        image and the aligned bounding box overlay to allow the user to verify
        low-confidence detections.
    *   **Custom Query Search Support**: The backend server supporting the Web
        UI must implement on-the-fly embedding extraction for custom HTTP/S
        query URIs by downloading the image, running the detector to identify
        target bounding boxes, preprocessing the crop, and capturing the
        embedding vector using the PyTorch forward hook on the classifier.
