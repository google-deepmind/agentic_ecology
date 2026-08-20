# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Mock server for a generic web UI backend to interact with vector databases.

This is a template server used as a reference for building lightweight
UIs to visualize, rank, filter, and annotate vector databases.
"""

import abc
import enum
import http.server
import json
import math
import os
import socketserver
import sys
import threading
import time
from typing import Any
import urllib.parse


class SearchMode(enum.StrEnum):
  QUERY = enum.auto()
  CLASSIFIER = enum.auto()


class AnnotationValue(enum.StrEnum):
  POSITIVE = enum.auto()
  NEGATIVE = enum.auto()
  UNCERTAIN = enum.auto()


class MediaType(enum.StrEnum):
  AUDIO = enum.auto()
  IMAGE = enum.auto()


def get_env_config() -> dict[str, Any]:
  """Retrieves, defaults, and validates environment variables."""
  port_str = os.environ.get("PORT", "8080")
  try:
    port = int(port_str)
  except ValueError:
    print(
        f"Error: Invalid PORT environment variable: {port_str!r}. "
        "Must be an integer.",
        file=sys.stderr,
    )
    sys.exit(1)

  workspace_dir = os.environ.get("WORKSPACE_DIR", "agent_workspace")
  media_dir = os.environ.get("MEDIA_DIR", os.path.join(workspace_dir, "media"))
  db_dir = os.environ.get("DB_DIR", os.path.join(workspace_dir, "databases"))

  return {
      "port": port,
      "workspace_dir": workspace_dir,
      "media_dir": media_dir,
      "db_dir": db_dir,
  }


def sanitize_float(val: Any) -> float | None:
  """Converts a value to float, returning None if NaN, Inf, or invalid.

  Args:
    val: The value to convert.

  Returns:
    The float value, or None if conversion failed or is non-finite.
  """
  if val is None:
    return None
  try:
    f = float(val)
    if math.isnan(f) or math.isinf(f):
      return None
    return f
  except (TypeError, ValueError):
    return None


class SafeEncoder(json.JSONEncoder):
  """JSON encoder that automatically converts NumPy scalars to Python types."""

  def default(self, o):
    if "numpy" in type(o).__module__ and hasattr(o, "item"):
      return o.item()
    return super().default(o)


# ==============================================================================
# SECTION 1: Database Adapter Abstraction
# ==============================================================================


class BaseDatabaseAdapter(abc.ABC):
  """Abstract interface for database operations used by the web server."""

  @abc.abstractmethod
  def list_databases(self) -> list[str]:
    """Returns a list of available database IDs."""

  @abc.abstractmethod
  def get_labels(self, database_id: str) -> list[str]:
    """Retrieves unique annotation labels present in the database.

    Args:
      database_id: The ID of the database to query.

    Returns:
      A list of unique annotation label strings.
    """

  @abc.abstractmethod
  def search(
      self, database_id: str, mode: SearchMode, label: str, query_uri: str
  ) -> dict[str, Any]:
    """Performs search/ranking and returns results matching Level 4 UI schema.

    Args:
      database_id: The ID of the database to search.
      mode: The search mode.
      label: The active annotation label to retrieve annotations for.
      query_uri: Optional URI of a query media file for similarity search.

    Returns:
        A dict with:
        {
          "query": {"uri": str|None, "media_type": str|None, "url": str|None},
          "results": list[dict],
          "classifier_trained": bool,
          "classifier_metrics": dict
        }
    """

  @abc.abstractmethod
  def annotate(
      self,
      database_id: str,
      item_id: int,
      label: str,
      annotation: AnnotationValue,
  ) -> None:
    """Submits or clears a user annotation.

    Args:
      database_id: The ID of the database.
      item_id: The numeric ID of the item to annotate.
      label: The annotation label name.
      annotation: The annotation value.
    """

  @abc.abstractmethod
  def train_classifier(self, database_id: str, label: str) -> dict[str, Any]:
    """Trains a classifier on user annotations and returns metrics.

    Args:
      database_id: The ID of the database.
      label: The annotation label name to train on.

    Returns:
      A dictionary of training evaluation metrics (e.g., accuracy, precision).
    """

  @abc.abstractmethod
  def get_media_stream(
      self, database_id: str, path: str, params: dict[str, list[str]]
  ) -> tuple[bytes, str]:
    """Returns the raw bytes of a media segment and its content type.

    Args:
      database_id: The ID of the database.
      path: The path or URI of the media file.
      params: Additional query parameters (e.g., dynamic window offsets).

    Returns:
        A tuple of (bytes, content_type).
    """

  @abc.abstractmethod
  def get_media_preview(
      self, database_id: str, item_id: int
  ) -> tuple[bytes, str]:
    """Generates and returns a visual/media preview for a database item.

    Args:
      database_id: The ID of the database.
      item_id: The ID of the database item to generate a preview for.

    Returns:
        A tuple of (bytes, content_type).
    """

  @abc.abstractmethod
  def get_query_preview(
      self, database_id: str, query_uri: str
  ) -> tuple[bytes, str]:
    """Generates and returns a visual/media preview for a query URI.

    Args:
      database_id: The ID of the database.
      query_uri: The URI of the query item to generate a preview for.

    Returns:
        A tuple of (bytes, content_type).
    """


# ==============================================================================
# SECTION 2: Default JSON Mock Adapter Implementation
# ==============================================================================


class JSONDatabaseAdapter(BaseDatabaseAdapter):
  """Mock implementation of BaseDatabaseAdapter using JSON database files."""

  def __init__(self, db_dir: str, media_dir: str):
    """Initializes the instance.

    Args:
      db_dir: The directory containing JSON database files.
      media_dir: The directory containing media files.
    """
    self.db_dir = db_dir
    self.media_dir = media_dir
    self.loaded_databases = {}
    self.db_locks = {}
    self.classifier_states = {}
    self.state_lock = threading.Lock()
    self.main_lock = threading.Lock()

  def _get_database(
      self, database_id: str
  ) -> tuple[dict[str, Any], threading.Lock]:
    """Loads and caches the requested database, returning data and lock.

    Args:
      database_id: The ID of the database to retrieve.

    Returns:
      A tuple of (database_data, database_lock) where database_data is the
        loaded JSON data and database_lock is a threading.Lock.

    Raises:
      FileNotFoundError: If the database JSON file does not exist.
    """
    database_id = os.path.basename(database_id)
    filepath = os.path.join(self.db_dir, f"{database_id}.json")

    if not os.path.exists(filepath):
      raise FileNotFoundError(f"Database {database_id} not found")

    with self.main_lock:
      if database_id not in self.db_locks:
        self.db_locks[database_id] = threading.Lock()

    with self.db_locks[database_id]:
      if database_id not in self.loaded_databases:
        with open(filepath, "r") as f:
          self.loaded_databases[database_id] = json.load(f)
      return self.loaded_databases[database_id], self.db_locks[database_id]

  def _save_database(self, database_id: str) -> None:
    """Saves the loaded database back to disk. Assumes database lock is held.

    Args:
      database_id: The ID of the database to save.
    """
    database_id = os.path.basename(database_id)
    filepath = os.path.join(self.db_dir, f"{database_id}.json")

    if database_id in self.loaded_databases:
      with open(filepath, "w") as f:
        json.dump(self.loaded_databases[database_id], f, indent=2)

  def list_databases(self) -> list[str]:
    try:
      if not os.path.exists(self.db_dir):
        return []
      files = os.listdir(self.db_dir)
      databases = [os.path.splitext(f)[0] for f in files if f.endswith(".json")]
      return sorted(databases)
    except Exception as e:
      raise RuntimeError(f"Failed to list databases: {e}") from e

  def get_labels(self, database_id: str) -> list[str]:
    db_data, db_lock = self._get_database(database_id)
    with db_lock:
      annotations = db_data.get("annotations", {})
      labels = set()
      for _, label_ann in annotations.items():
        for label in label_ann.keys():
          labels.add(label)
    return sorted(list(labels))

  def search(
      self, database_id: str, mode: SearchMode, label: str, query_uri: str
  ) -> dict[str, Any]:
    db_data, db_lock = self._get_database(database_id)
    classifier_key = (database_id, label)

    if mode == SearchMode.CLASSIFIER:
      with self.state_lock:
        class_state = self.classifier_states.get(classifier_key)
      if not class_state or not class_state.get("trained"):
        raise ValueError(
            f"Model not trained yet for database '{database_id}' and label"
            f" '{label}'."
        )

      results_copy = []
      with db_lock:
        annotations = db_data.get("annotations", {})
        for r in class_state["results"]:
          r_copy = dict(r)
          r_copy["annotation"] = annotations.get(str(r["id"]), {}).get(
              label, AnnotationValue.UNCERTAIN
          )
          results_copy.append(r_copy)

      return {
          "query": {"uri": None, "media_type": None, "url": None},
          "results": results_copy,
          "classifier_metrics": class_state["metrics"],
          "classifier_trained": True,
      }
    else:
      results_copy = []
      with db_lock:
        annotations = db_data.get("annotations", {})
        for item in db_data["items"]:
          item_id_str = str(item["id"])
          annotation = annotations.get(item_id_str, {}).get(
              label, AnnotationValue.UNCERTAIN
          )
          score = item["query_sim"] if query_uri else 0.0
          results_copy.append({
              "id": item["id"],
              "title": item["title"],
              "media_type": item["media_type"],
              "media_url": item["media_url"],
              "score": sanitize_float(score) or 0.0,
              "annotation": annotation,
          })

      if query_uri:
        results_copy.sort(key=lambda x: x["score"], reverse=True)

      with self.state_lock:
        class_state = self.classifier_states.get(classifier_key)
        classifier_trained = (
            class_state.get("trained", False) if class_state else False
        )
        classifier_metrics = (
            class_state.get("metrics", {}) if class_state else {}
        )

      return {
          "query": {
              "uri": query_uri if query_uri else None,
              "media_type": "audio" if query_uri else None,
              "url": (
                  f"/media/query_ref.wav?database={urllib.parse.quote(database_id)}&path={urllib.parse.quote(query_uri)}"
                  if query_uri
                  else None
              ),
          },
          "results": results_copy,
          "classifier_trained": classifier_trained,
          "classifier_metrics": classifier_metrics,
      }

  def annotate(
      self,
      database_id: str,
      item_id: int,
      label: str,
      annotation: AnnotationValue,
  ) -> None:
    db_data, db_lock = self._get_database(database_id)
    with db_lock:
      annotations = db_data.setdefault("annotations", {})
      item_id_str = str(item_id)
      if annotation == AnnotationValue.UNCERTAIN:
        if item_id_str in annotations and label in annotations[item_id_str]:
          del annotations[item_id_str][label]
          if not annotations[item_id_str]:
            del annotations[item_id_str]
      else:
        annotations.setdefault(item_id_str, {})[label] = annotation

      self._save_database(database_id)

  def train_classifier(self, database_id: str, label: str) -> dict[str, Any]:
    db_data, db_lock = self._get_database(database_id)
    with db_lock:
      annotations = db_data.get("annotations", {})
      positives = 0
      negatives = 0
      for _, label_ann in annotations.items():
        val = label_ann.get(label)
        if val == AnnotationValue.POSITIVE:
          positives += 1
        elif val == AnnotationValue.NEGATIVE:
          negatives += 1

    total_annotated = positives + negatives
    if positives < 1 or negatives < 1 or total_annotated < 3:
      raise ValueError(
          "Please label at least 3 items (mix of 👍 and 👎) for label"
          f" '{label}'"
          " to train."
      )

    time.sleep(1.2)  # Simulate ML training delay

    metrics = {
        "accuracy": sanitize_float(0.85 + (positives * 0.02)),
        "roc_auc": sanitize_float(0.89 + (positives * 0.015)),
        "precision": sanitize_float(0.82 + (positives * 0.02)),
        "dataset_size": int(total_annotated),
    }

    serialized_results = []
    with db_lock:
      for item in db_data["items"]:
        item_id_str = str(item["id"])
        annotation = annotations.get(item_id_str, {}).get(
            label, AnnotationValue.UNCERTAIN
        )
        if annotation == AnnotationValue.POSITIVE:
          score = 0.85 + (item["id"] * 0.001)
        elif annotation == AnnotationValue.NEGATIVE:
          score = 0.10 - (item["id"] * 0.001)
        else:
          score = 0.35 + ((item["id"] % 5) * 0.08)

        serialized_results.append({
            "id": item["id"],
            "title": item["title"],
            "media_type": item["media_type"],
            "media_url": item["media_url"],
            "score": sanitize_float(score) or 0.0,
            "annotation": annotation,
        })
    serialized_results.sort(key=lambda x: x["score"], reverse=True)

    classifier_key = (database_id, label)
    with self.state_lock:
      self.classifier_states[classifier_key] = {
          "trained": True,
          "metrics": metrics,
          "results": serialized_results,
      }

    return metrics

  def get_media_stream(
      self, database_id: str, path: str, params: dict[str, list[str]]
  ) -> tuple[bytes, str]:
    filename = os.path.basename(path)
    filepath = os.path.join(self.media_dir, filename)

    if not os.path.exists(filepath):
      b = b"RIFF_MOCK_MEDIA_FILE_BYTES"
      content_type = "audio/wav" if filename.endswith(".wav") else "image/jpeg"
      return b, content_type

    with open(filepath, "rb") as f:
      b = f.read()
    content_type = "audio/wav" if filename.endswith(".wav") else "image/jpeg"
    return b, content_type

  def get_media_preview(
      self, database_id: str, item_id: int
  ) -> tuple[bytes, str]:
    # Return a 1x1 transparent PNG
    return (
        (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
            b"\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00"
            b"\x01\x00\x00\x0c\x00\x01\x04q\x07\xdf\x00\x00\x00\x00IEND\xaeB`"
            b"\x82"
        ),
        "image/png",
    )

  def get_query_preview(
      self, database_id: str, query_uri: str
  ) -> tuple[bytes, str]:
    # Return a 1x1 transparent PNG
    return (
        (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
            b"\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00"
            b"\x01\x00\x00\x0c\x00\x01\x04q\x07\xdf\x00\x00\x00\x00IEND\xaeB`"
            b"\x82"
        ),
        "image/png",
    )


# Global Active Database Adapter Reference (initialized at server startup)
ACTIVE_ADAPTER: BaseDatabaseAdapter | None = None


def get_adapter() -> BaseDatabaseAdapter:
  """Retrieves the active database adapter, raising an error if uninitialized.

  Returns:
    The active BaseDatabaseAdapter instance.

  Raises:
    RuntimeError: If the adapter has not been initialized.
  """
  if ACTIVE_ADAPTER is None:
    raise RuntimeError(
        "ACTIVE_ADAPTER is not initialized. "
        "Call start_server() or init_adapter() first."
    )
  return ACTIVE_ADAPTER


def init_adapter(adapter: BaseDatabaseAdapter):
  """Allows replacing the active database adapter programmatically.

  Args:
    adapter: The database adapter instance to use.
  """
  global ACTIVE_ADAPTER
  ACTIVE_ADAPTER = adapter


# ==============================================================================
# SECTION 3: HTTP Server Router
# ==============================================================================


class Handler(http.server.SimpleHTTPRequestHandler):
  """HTTP Handler for serving static files and API endpoints."""

  # Configured dynamically at runtime
  workspace_dir = "agent_workspace"

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    """Initializes the handler instance."""
    super().__init__(*args, directory=self.workspace_dir, **kwargs)

  def do_GET(self) -> None:
    """Handles HTTP GET requests."""
    parsed = urllib.parse.urlparse(self.path)
    params = urllib.parse.parse_qs(parsed.query)

    match parsed.path:
      case "/":
        self.path = "/index.html"
        super().do_GET()
      case "/api/databases":
        self.handle_get_databases()
      case "/api/labels":
        self.handle_get_labels(params)
      case "/api/search":
        self.handle_get_search(params)
      case "/api/preview":
        self.handle_get_preview(params)
      case "/stream":
        self.handle_media_stream(parsed.path, params)
      case path if path.startswith("/media/"):
        self.handle_media_stream(parsed.path, params)
      case _:
        super().do_GET()

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    """Handles HTTP POST requests."""
    parsed = urllib.parse.urlparse(self.path)
    params = urllib.parse.parse_qs(parsed.query)
    match parsed.path:
      case "/api/annotation":
        self.handle_post_annotation()
      case "/api/train":
        self.handle_post_train(params)
      case _:
        self.send_error(404)

  def handle_get_databases(self) -> None:
    """Lists available databases."""
    try:
      databases = get_adapter().list_databases()
      self.send_json({"databases": databases})
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_json_error(500, str(e))

  def handle_get_labels(self, params: dict[str, list[str]]) -> None:
    """Retrieves unique annotation labels present in the database.

    Args:
      params: Dict of query parameters containing the database name.
    """
    database_id = params.get("database", [""])[0]
    if not database_id:
      self.send_json_error(400, "Missing 'database' parameter.")
      return
    try:
      labels = get_adapter().get_labels(database_id)
      self.send_json({"labels": labels})
    except FileNotFoundError:
      self.send_json_error(404, f"Database '{database_id}' not found.")
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_json_error(500, str(e))

  def handle_get_search(self, params: dict[str, list[str]]) -> None:
    """Handles GET requests for search results with dynamic label support.

    Args:
      params: Dict of query parameters containing query config.
    """
    database_id = params.get("database", [""])[0]
    mode_str = params.get("mode", ["query"])[0]
    label = params.get("label", ["default_label"])[0]
    query_uri = params.get("query_uri", [""])[0].strip()

    if not database_id:
      self.send_json_error(400, "Missing 'database' parameter.")
      return

    try:
      mode = SearchMode(mode_str)
    except ValueError:
      self.send_json_error(400, f"Invalid 'mode' parameter: {mode_str}")
      return

    try:
      response_data = get_adapter().search(
          database_id=database_id, mode=mode, label=label, query_uri=query_uri
      )
      self.send_json(response_data)
    except FileNotFoundError:
      self.send_json_error(404, f"Database '{database_id}' not found.")
    except ValueError as ve:
      self.send_json_error(400, str(ve))
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_json_error(500, str(e))

  def handle_get_preview(self, params: dict[str, list[str]]) -> None:
    """Generates and serves a visual/media preview for a specific item.

    Args:
      params: Dict of query parameters containing database and item IDs.
    """
    database_id = params.get("database", [""])[0]
    query_uri = params.get("query_uri", [""])[0]
    item_id_str = params.get("item_id", params.get("window_id", [""]))[0]

    if not database_id:
      self.send_json_error(400, "Missing 'database' parameter.")
      return

    error_item = query_uri or item_id_str
    try:
      if query_uri:
        img_bytes, content_type = get_adapter().get_query_preview(
            database_id, query_uri
        )
      elif item_id_str:
        try:
          item_id = int(item_id_str)
        except ValueError:
          self.send_json_error(400, "Invalid 'item_id' parameter.")
          return
        img_bytes, content_type = get_adapter().get_media_preview(
            database_id, item_id
        )
      else:
        self.send_json_error(400, "Missing 'item_id' or 'query_uri' parameter.")
        return

      self.send_response(200)
      self.send_header("Content-type", content_type)
      self.send_header("Content-Length", str(len(img_bytes)))
      self.send_header("Cache-Control", "public, max-age=3600")
      self.end_headers()
      self.wfile.write(img_bytes)
    except FileNotFoundError:
      self.send_json_error(
          404, f"Database '{database_id}' or item '{error_item}' not found."
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_json_error(500, str(e))

  def handle_post_annotation(self) -> None:
    """Handles POST requests for submitting annotation."""
    content_length = int(self.headers["Content-Length"])
    post_data = self.rfile.read(content_length)
    try:
      data = json.loads(post_data.decode("utf-8"))
      database_id = data.get("database")
      item_id = int(data["item_id"])
      annotation_str = data["annotation"]
      label = data.get("label", "default_label")

      if not database_id:
        self.send_json_error(400, "Missing 'database' parameter.")
        return

      try:
        annotation = AnnotationValue(annotation_str)
      except ValueError:
        self.send_json_error(
            400, f"Invalid 'annotation' parameter: {annotation_str}"
        )
        return

      get_adapter().annotate(
          database_id=database_id,
          item_id=item_id,
          label=label,
          annotation=annotation,
      )
      self.send_json({"status": "success"})
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_json_error(500, str(e))

  def handle_post_train(self, params: dict[str, list[str]]) -> None:
    """Handles POST requests for training the classifier.

    Args:
      params: Dict of query parameters containing database and label names.
    """
    database_id = params.get("database", [""])[0]
    label = params.get("label", ["default_label"])[0]

    if not database_id:
      self.send_json_error(400, "Missing 'database' parameter.")
      return

    try:
      metrics = get_adapter().train_classifier(
          database_id=database_id, label=label
      )
      self.send_json({"status": "success", "metrics": metrics})
    except ValueError as ve:
      self.send_json_error(400, str(ve))
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_json_error(500, f"Training failed: {e}")

  def handle_media_stream(
      self, request_path: str, params: dict[str, list[str]]
  ) -> None:
    """Handles streaming media chunk/range requests by calling adapter.

    Args:
      request_path: The request path of the media stream.
      params: Dict of query parameters.
    """
    # Determine the target database and media path/URI
    database_id = params.get("database", [""])[0]

    # For index.html requests:
    # Under standard mock it uses: /media/item_1.wav
    # Under custom endpoints it might use: /stream?database=...&window_id=...
    path = request_path
    if request_path.startswith("/media/"):
      path = request_path[7:]  # Strip '/media/'

    try:
      b, content_type = get_adapter().get_media_stream(
          database_id=database_id, path=path, params=params
      )
      self.send_response(200)
      self.send_header("Content-Type", content_type)
      self.send_header("Content-Length", str(len(b)))
      self.end_headers()
      self.wfile.write(b)
    except (BrokenPipeError, ConnectionResetError):
      pass
    except Exception as e:  # pylint: disable=broad-exception-caught
      self.send_error(500, f"Streaming error: {e}")

  def send_json(self, data: Any, status_code: int = 200) -> None:
    """Helper method to serialize data to JSON and send HTTP response.

    Args:
      data: The Python data structure to serialize.
      status_code: The HTTP status code to return.
    """
    b = json.dumps(data, cls=SafeEncoder).encode("utf-8")
    self.send_response(status_code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(b)))
    self.end_headers()
    try:
      self.wfile.write(b)
    except (BrokenPipeError, ConnectionResetError):
      pass

  def send_json_error(self, code: int, message: str) -> None:
    """Helper method to send a JSON-formatted error response.

    Args:
      code: The HTTP error status code.
      message: The error message string.
    """
    self.send_json({"error": message}, status_code=code)


def create_mock_database(filepath: str, media_type: MediaType) -> None:
  """Creates a mock database JSON file for testing.

  Args:
    filepath: Path to the destination JSON file.
    media_type: Type of media, e.g., "audio" or "image".
  """
  items = []
  name_prefix = (
      "Audio Clip" if media_type == MediaType.AUDIO else "Camera Trap Photo"
  )
  ext = "wav" if media_type == MediaType.AUDIO else "jpg"

  for i in range(1, 101):
    items.append({
        "id": i,
        "title": f"{name_prefix} #{i}",
        "media_type": media_type.value,
        "media_url": f"/media/item_{i}.{ext}",
        "query_sim": 1.0 - (i * 0.0095),
    })

  if media_type == MediaType.AUDIO:
    annotations = {
        "5": {
            "bird_call": AnnotationValue.POSITIVE,
            "wind_noise": AnnotationValue.NEGATIVE,
        },
        "12": {"bird_call": AnnotationValue.NEGATIVE},
        "20": {
            "bird_call": AnnotationValue.POSITIVE,
            "wind_noise": AnnotationValue.POSITIVE,
        },
    }
  else:
    annotations = {
        "3": {
            "jaguar": AnnotationValue.POSITIVE,
            "empty": AnnotationValue.NEGATIVE,
        },
        "8": {"jaguar": AnnotationValue.NEGATIVE},
        "15": {
            "jaguar": AnnotationValue.POSITIVE,
            "empty": AnnotationValue.POSITIVE,
        },
    }

  db_data = {"items": items, "annotations": annotations}

  with open(filepath, "w") as f:
    json.dump(db_data, f, indent=2)


def init_databases(db_dir: str) -> None:
  """Initializes the database directory and default mock databases.

  Args:
    db_dir: Path to the database directory.
  """
  os.makedirs(db_dir, exist_ok=True)

  bio_db_path = os.path.join(db_dir, "bioacoustics.json")
  if not os.path.exists(bio_db_path):
    print("Creating mock bioacoustics database...")
    create_mock_database(bio_db_path, MediaType.AUDIO)

  cam_db_path = os.path.join(db_dir, "camera_traps.json")
  if not os.path.exists(cam_db_path):
    print("Creating mock camera traps database...")
    create_mock_database(cam_db_path, MediaType.IMAGE)


def start_server() -> None:
  """Starts the mock HTTP server using environment configuration."""
  global ACTIVE_ADAPTER
  config = get_env_config()

  # Configure Handler workspace directory dynamically
  Handler.workspace_dir = config["workspace_dir"]

  # Initialize default adapter if not already programmatically configured
  if ACTIVE_ADAPTER is None:
    ACTIVE_ADAPTER = JSONDatabaseAdapter(config["db_dir"], config["media_dir"])

  init_databases(config["db_dir"])

  socketserver.ThreadingTCPServer.allow_reuse_address = True
  with socketserver.ThreadingTCPServer(("", config["port"]), Handler) as httpd:
    print(f"Server running on port {config['port']}")
    httpd.serve_forever()


if __name__ == "__main__":
  start_server()
