# SpeciesNet Technical Reference

Detailed instructions and code examples for performing object detection,
classification, and embedding extraction using `speciesnet` and saving them into
a `perch-hoplite` database.

## Script Entry Point & Safety Imports

Camera trap ingestion scripts load PyTorch (via `yolov5` in SpeciesNet), TensorFlow,
and Perch Hoplite (PyArrow). Adhere to the safety import sequence and local cache definitions:

```python
import os
import pathlib

# Configure local caches for sandboxed or headless environments
os.environ.setdefault("KERAS_HOME", str(pathlib.Path.cwd() / ".keras"))
os.environ.setdefault("KAGGLEHUB_CACHE", str(pathlib.Path.cwd() / ".cache" / "kagglehub"))

# Rule 1b: yolov5 must precede tensorflow to prevent segmentation fault
import yolov5  # noqa: F401 # isort: skip

# Rule 1: tensorflow must precede perch_hoplite / pyarrow to prevent macOS Abseil deadlock
import tensorflow as tf  # noqa: F401 # isort: skip

import logging


class SQLSuppressFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Executed SQL statement" not in record.getMessage()


logging.getLogger("absl").addFilter(SQLSuppressFilter())
```

## Model Ingestion Setup

### 1. Database Configuration

Hoplite's database must be initialized to support 1280-dimensional vectors using
a `float16` scalar kind and the `Cos` metric.

```python
from ml_collections import config_dict
from perch_hoplite.db import sqlite_usearch_impl

usearch_cfg = config_dict.ConfigDict(
    {
        "embedding_dim": 1280,
        "metric_name": "Cos",
        "expansion_add": 16,
        "expansion_search": 16,
        "dtype": "float16",  # Crucial: perch-hoplite only supports float16 in USEARCH_DTYPES
    }
)
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

#### Handling Read-Only Filesystem Mounts (Colab / Kaggle Environments)

- **Issue:** When running inside an environment with pre-mounted Kaggle models
  (such as Colab VMs or Kaggle environments where `kagglehub` resolves to
  `/kaggle/input/`), the target directory is mounted as read-only. Because
  model loaders (such as `SpeciesNetDetector` and `SpeciesNetClassifier`)
  attempt to download and write additional weights or configurations into the
  model folder, initializing them directly against `/kaggle/input/...` fails
  with an `OSError: [Errno 30] Read-only file system`.
- **Workaround:** Copy the mounted model directory from `/kaggle/input/...` to
  a local writable directory (such as `/content/speciesnet_model` on Colab or
  `/tmp/speciesnet_model`) prior to instantiating the detector and classifier:

```python
import os
import shutil

local_model_dir = "/content/speciesnet_model"
mounted_model_dir = "/kaggle/input/speciesnet/pytorch/v4.0.3a/1"
if os.path.exists(mounted_model_dir) and not os.path.exists(local_model_dir):
    shutil.copytree(mounted_model_dir, local_model_dir)

detector = SpeciesNetDetector(local_model_dir)
classifier = SpeciesNetClassifier(local_model_dir)
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

MegaDetector returns normalized relative coordinates: `[xmin, ymin, width, height]`.
Clamp coordinates to ensure valid PIL crop boundaries:

```python
left = max(0, min(img.width - 1, int(bbox[0] * img.width)))
top = max(0, min(img.height - 1, int(bbox[1] * img.height)))
right = max(left + 1, min(img.width, int((bbox[0] + bbox[2]) * img.width)))
bottom = max(top + 1, min(img.height, int((bbox[1] + bbox[3]) * img.height)))

crop_img = img.crop((left, top, right, bottom))
```

### 5. Ingestion Insertions

Link images to a deployment and add detected animal crops as windows containing
their spatial offsets:

```python
deployment_id = db.insert_deployment(name="KGA_S1", project="KGA")

# For each image:
recording_id = db.insert_recording(
    filename=rel_path, datetime=None, deployment_id=deployment_id
)

# For each detection:
db.insert_window(
    recording_id=recording_id,
    offsets=[
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[0] + bbox[2]),
        float(bbox[1] + bbox[3]),
    ],  # Store relative bounding box coordinates
    embedding=embedding_vector,
    handle_duplicates="allow",
)

# CRITICAL: Always commit at the end of ingestion (and periodically during large batches).
# db.commit() commits the SQLite transaction AND flushes the USearch index to disk (usearch.index).
db.commit()
```

## Custom Image Similarity Search (On-the-Fly Embedding)

When the user enters a custom query URI (such as an external HTTP/S image URL or
a local file path) instead of an existing database window ID, you must extract
its embedding on-the-fly:

1. **Resolve and Load Image**: Download the image (for HTTP/S URLs) using
   `urllib.request` or load it from disk, and convert it to RGB format.
1. **Detect Bounding Box**: Run the preloaded `SpeciesNetDetector` model on the
   image. If detections are found, extract the highest-confidence bounding box.
1. **Crop and Classify**: Pass the image and the bounding box to
   `SpeciesNetClassifier.preprocess(img, bboxes=[bbox_obj])`.
1. **Hook Embedding**: Intercept the average pooling layer of the classifier
   during prediction using the forward hook to extract the 1280-dimensional
   feature vector, cast it to `float16`, and run the USearch similarity search.
1. **Caching**: Cache both the query preview image and the extracted embedding
   to prevent duplicate downloads and model inference passes on page refreshes.

## Querying & Neighbor Lookup

When querying the USearch index via `db.search()`, neighbor keys (`match.window_id`)
are returned as NumPy integers (`numpy.uint64`). Standard Python `sqlite3` parameterized
queries (`WHERE id = ?`) do not match NumPy scalar types against integer columns.

Always cast the window ID to Python's native `int`:

```python
for match in search_results:
    window_id = int(match.window_id)  # Crucial: cast numpy.uint64 to native int
    cursor = db._get_cursor()
    row = cursor.execute(
        "SELECT w.id, r.filename, w.offsets FROM windows w JOIN recordings r ON w.recording_id = r.id WHERE w.id = ?",
        (window_id,)
    ).fetchone()
```
