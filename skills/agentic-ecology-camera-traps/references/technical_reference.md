# SpeciesNet Technical Reference

Detailed instructions and code examples for performing object detection,
classification, and embedding extraction using `speciesnet` and saving them into
a `perch-hoplite` database.

## Model Ingestion Setup

### 1. Database Configuration

Hoplite's database must be initialized to support 1280-dimensional vectors using
a `float16` scalar kind and the `Cos` metric.

```python
from ml_collections import config_dict
from perch_hoplite.db import sqlite_usearch_impl

usearch_cfg = config_dict.ConfigDict({
    "embedding_dim": 1280,
    "metric_name": "Cos",
    "expansion_add": 16,
    "expansion_search": 16,
    "dtype": "float16" # Crucial: perch-hoplite only supports float16 in USEARCH_DTYPES
})
db = sqlite_usearch_impl.SQLiteUSearchDB.create("databases/kga_hoplite", usearch_cfg)
```

### 2. Loading SpeciesNet Models

Load both detector and classifier from Kaggle or HuggingFace identifiers:

```python
from speciesnet.detector import SpeciesNetDetector
from speciesnet.classifier import SpeciesNetClassifier

model_name = "kaggle:google/speciesnet/pyTorch/v4.0.3a/1"
detector = SpeciesNetDetector(model_name)
classifier = SpeciesNetClassifier(model_name)
```

### 3. Extracting Embeddings via PyTorch Hook

Hook the squeeze layer immediately preceding the linear classification head:

```python
import numpy as np

# Find target module
modules_dict = dict(classifier.model.named_modules())
target_layer = modules_dict["SpeciesNet/efficientnetv2-m/avg_pool/Mean_Squeeze__3825"]

# Define hook callback
captured_embeddings = []
def hook_fn(module, input_tensor, output_tensor):
    captured_embeddings.append(output_tensor.cpu().numpy().squeeze())

# Register hook
hook_handle = target_layer.register_forward_hook(hook_fn)

# After running predictor, target embedding is captured
captured_embeddings.clear()
_ = classifier.predict("crop.jpg", preprocessed_img)

if captured_embeddings:
    # Cast to float16 to match USearch index data type
    embedding_vector = captured_embeddings[0].astype(np.float16)
```

### 4. Crop Bounding Box Calculations

MegaDetector returns normalized relative coordinates: `[xmin, ymin, width,
height]`. Convert them to absolute pixel coordinates for PIL cropping:

```python
left = int(bbox[0] * img.width)
top = int(bbox[1] * img.height)
right = int((bbox[0] + bbox[2]) * img.width)
bottom = int((bbox[1] + bbox[3]) * img.height)

crop_img = img.crop((left, top, right, bottom))
```

### 5. Ingestion Insertions

Link images to a deployment and add detected animal crops as windows containing
their spatial offsets:

```python
deployment_id = db.insert_deployment(name="KGA_S1", project="KGA")

# For each image:
recording_id = db.insert_recording(
    filename=rel_path,
    datetime=None,
    deployment_id=deployment_id
)

# For each detection:
db.insert_window(
    recording_id=recording_id,
    offsets=[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], # Store relative bounding box coordinates
    embedding=embedding_vector,
    handle_duplicates="allow"
)
```

## Custom Image Similarity Search (On-the-Fly Embedding)

When the user enters a custom query URI (such as an external HTTP/S image URL or
a local file path) instead of an existing database window ID, you must extract
its embedding on-the-fly:

1.  **Resolve and Load Image**: Download the image (for HTTP/S URLs) using
    `urllib.request` or load it from disk, and convert it to RGB format.
2.  **Detect Bounding Box**: Run the preloaded `SpeciesNetDetector` model on the
    image. If detections are found, extract the highest-confidence bounding box.
3.  **Crop and Classify**: Pass the image and the bounding box to
    `SpeciesNetClassifier.preprocess(img, bboxes=[bbox_obj])`.
4.  **Hook Embedding**: Intercept the average pooling layer of the classifier
    during prediction using the forward hook to extract the 1280-dimensional
    feature vector, cast it to `float16`, and run the USearch similarity search.
5.  **Caching**: Cache both the query preview image and the extracted embedding
    to prevent duplicate downloads and model inference passes on page refreshes.
