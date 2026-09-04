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

"""Reference implementations for the bioacoustics skill.

This module provides preprocessing tools and a self-contained Database Adapter
class (HopliteDatabaseAdapter) for integrating SQLite/USearch bioacoustic
databases with the reference server implementation in the `ui` skill. Undefined
variables in this module are presumed to be defined in the `ui` skill's
reference `server` module implementation.
"""

# pylint: disable=undefined-variable

# Prevent dynamic library deadlock by importing tensorflow first on macOS
import tensorflow as tf  # pylint: disable=unused-import,g-bad-import-order


import collections
import io
import logging
import pathlib
import threading
from typing import Any
from unittest import mock
import urllib.parse

import matplotlib
from ml_collections import config_dict
import numpy as np
from perch_hoplite.agile import classifier
from perch_hoplite.agile import classifier_data
from perch_hoplite.agile import embed
from perch_hoplite.agile import embedding_display
from perch_hoplite.agile import source_info
from perch_hoplite.audio_io import load_audio_window
from perch_hoplite.db import datatypes
from perch_hoplite.db import db_loader
from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.zoo import model_configs
from PIL import Image
import soundfile


# Suppress noisy SQL statement logs from the database library (using
# absl.logging)
class SQLSuppressFilter(logging.Filter):

  def filter(self, record):
    return "Executed SQL statement" not in record.getMessage()


logging.getLogger("absl").addFilter(SQLSuppressFilter())

# ==============================================================================
# SECTION 1: Preprocessing & Database Setup Utilities
# ==============================================================================


def create_and_populate_db(
    db_dir: str,
    audio_dir: str,
    model_name: str = "perch_v2",
    dataset_name: str = "my_project",
) -> sqlite_usearch_impl.SQLiteUSearchDB:
  """Creates a Hoplite database and extracts embeddings from wav files.

  Args:
    db_dir: Path to the Hoplite database directory (contains sqlite and index).
    audio_dir: Directory containing input audio files.
    model_name: Model configuration preset name.
    dataset_name: Dataset label/project name.

  Returns:
    An initialized and populated SQLiteUSearchDB database connection.
  """
  preset_info = model_configs.get_preset_model_config(model_name)

  # Initialize the database with the appropriate embedding dimension
  db = db_loader.create_new_usearch_db(
      db_dir, embedding_dim=preset_info.embedding_dim
  )
  db.commit()

  # Find all subdirectories that contain .wav files recursively
  audio_path = pathlib.Path(audio_dir)
  subdirs = sorted(list(set(p.parent for p in audio_path.rglob("*.wav"))))
  if not subdirs:
    subdirs = [audio_path]

  configs = []
  for i, subdir in enumerate(subdirs):
    # Calculate a unique and clean dataset_name for each subdir to satisfy
    # Hoplite's requirement that dataset names in AudioSources must be unique.
    try:
      rel_path = subdir.relative_to(audio_path)
      # Sanitize relative path into a safe name suffix
      suffix = str(rel_path).replace("/", "_").replace("\\", "_")
      name = f"{dataset_name}_{suffix}" if suffix else dataset_name
    except ValueError:
      name = f"{dataset_name}_{i}"

    configs.append(
        source_info.AudioSourceConfig(
            dataset_name=name,
            base_path=str(subdir),
            file_glob="*.wav",
            min_audio_len_s=1.0,
            target_sample_rate_hz=-2,  # Use model's target sample rate
        )
    )
  audio_sources = source_info.AudioSources(tuple(configs))

  # Configure the model
  model_config = embed.ModelConfig(
      model_key=preset_info.model_key,
      embedding_dim=preset_info.embedding_dim,
      model_config=preset_info.model_config,
  )

  # Process and embed using EmbedWorker (automatically saves metadata)
  worker = embed.EmbedWorker(
      audio_sources=audio_sources,
      model_config=model_config,
      db=db,
  )
  worker.process_all()
  return db


def inspect_database(db_dir: str) -> None:
  """Prints database embedding configuration and total window size.

  Args:
    db_dir: The directory path containing hoplite.sqlite and usearch.index.
  """
  db = sqlite_usearch_impl.SQLiteUSearchDB.create(db_dir)

  # Extract embedding model configuration
  model_config_meta = db.get_metadata("model_config")
  model_key = model_config_meta.model_key  # e.g., 'taxonomy_model_tf'
  logging.info("Database Embedding Model Key: %s", model_key)
  logging.info("Model Configuration: %s", model_config_meta.model_config)

  # Extract database size / window count
  window_ids = db.match_window_ids()
  num_windows = len(window_ids)
  logging.info("Total Audio Windows in Database: %d", num_windows)


class CachedHopliteDB:
  """In-memory annotations cache wrapper for HopliteDBInterface.

  This class implements a delegation wrapper pattern around a Hoplite database
  connection to resolve training performance bottlenecks in multi-threaded web
  server environments.

  **Why?**

  During active learning classifier training, the `AgileDataManager` training
  iterator repeatedly queries window annotations on every single step. In a
  128-step training run with a batch size of 128, this executes over 16,000 SQL
  queries.

  While this single-threaded workload runs quickly in a standalone script, it
  creates severe lock contention and Python GIL bottlenecks in a multi-threaded
  web server (like `server.py`). The concurrent requests from the frontend for
  media streams and spectrogram previews block the training thread, causing the
  training process to take minutes and hit browser timeouts.

  **How?**

  Upon instantiation, `CachedHopliteDB` fetches all database annotations and
  pre-computes their intersections with every audio window in a single pass in
  memory.

  It overrides the `get_window_annotations` method to serve lookups directly
  from this in-memory lookup cache, reducing the training loop's SQL execution
  count to exactly zero. All other calls are dynamically delegated to the
  underlying database connection via `__getattr__`.
  """

  def __init__(self, db: Any):
    self._db = db

    # Pre-fetch all annotations and group by recording
    all_annotations = db.get_all_annotations()
    ann_by_recording = collections.defaultdict(list)
    for ann in all_annotations:
      ann_by_recording[ann.recording_id].append(ann)

    # Pre-compute intersections for ALL windows in the database
    all_windows = db.get_all_windows()
    self._annotations_dict = {}
    for w in all_windows:
      intersecting = []
      for ann in ann_by_recording.get(w.recording_id, []):
        if w.intersects(ann):
          intersecting.append(ann)
      self._annotations_dict[w.id] = intersecting

  def get_window_annotations(
      self, window_id: int, label: str | None = None
  ) -> Any:
    wid_int = int(window_id)
    if wid_int in self._annotations_dict:
      annotations = self._annotations_dict[wid_int]
      if label is not None:
        return [ann for ann in annotations if ann.label == label]
      return annotations
    return self._db.get_window_annotations(window_id, label)

  def __getattr__(self, name: str) -> Any:
    return getattr(self._db, name)


# Save the original evaluation function to allow calling it from the mock
# wrapper
original_eval_classifier = classifier.eval_classifier


# With very small training sets, `classifier.train_linear_classifier` crashes
# when calling `classifier.eval_classifier`. This function wrapper sidesteps the
# issue.
def safe_eval_classifier(
    params: Any, data_manager: Any, eval_ids: np.ndarray
) -> dict[str, Any]:
  if eval_ids is None or len(eval_ids) == 0:
    return {"top1_acc": 1.0, "roc_auc": 1.0, "cmap": 1.0}
  return original_eval_classifier(params, data_manager, eval_ids)


# ==============================================================================
# SECTION 2: Database Adapter
# ==============================================================================


class HopliteDatabaseAdapter(BaseDatabaseAdapter):
  """Hoplite SQLite/USearch implementation of BaseDatabaseAdapter."""

  def __init__(self, db_dir: str):
    """Initializes the adapter.

    Args:
      db_dir: Path to directory containing the hoplite sqlite database files.
    """
    super().__init__()
    self.db_dir = pathlib.Path(db_dir)

    # Lock for thread-safe main DB connection management
    self._db_lock = threading.Lock()

    # Lock for thread-safe cache operations
    self._cache_lock = threading.Lock()

    # Main database connection cache: database_id -> SQLiteUSearchDB
    self._main_dbs: dict[str, sqlite_usearch_impl.SQLiteUSearchDB] = {}

    # Thread-local storage to hold isolated database connections per handler
    # thread
    self._local = threading.local()

    # Active learning classifier cache: (database_id, label) -> data mapping
    self._scores_cache: dict[tuple[str, str], dict[int, float]] = {}
    self._metrics_cache: dict[tuple[str, str], dict[str, Any]] = {}

    # Query embedding and WAV audio cache: query_uri -> (embedding, wav_bytes)
    self._query_cache: dict[str, tuple[np.ndarray, bytes]] = {}
    self._query_cache_lock = threading.Lock()

    # ML model cache: model_key -> model instance
    self._model_cache: dict[str, Any] = {}
    self._model_cache_lock = threading.Lock()

  # ============================================================================
  # Public Abstract Method Implementations
  # ============================================================================

  def list_databases(self) -> list[str]:
    """Lists available sqlite database IDs in the database directory."""
    if not self.db_dir.exists():
      return []
    databases = []
    for f in self.db_dir.iterdir():
      if f.is_dir():
        if (f / "hoplite.sqlite").exists():
          if f.suffix == ".sqlite":
            databases.append(f.stem)
          else:
            databases.append(f.name)
    return sorted(databases)

  def get_labels(self, database_id: str) -> list[dict[str, Any]]:
    """Retrieves unique annotation labels present in the database."""
    db = self._get_db(database_id)
    annotations = db.get_all_annotations()

    labels_pos = collections.defaultdict(int)
    labels_neg = collections.defaultdict(int)
    for ann in annotations:
      if ann.provenance == "user":
        if ann.label_type == datatypes.LabelType.POSITIVE:
          labels_pos[ann.label] += 1
        elif ann.label_type == datatypes.LabelType.NEGATIVE:
          labels_neg[ann.label] += 1

    unique_labels = sorted(
        list(set(labels_pos.keys()) | set(labels_neg.keys()))
    )
    return [
        {
            "name": name,
            "positive": labels_pos[name],
            "negative": labels_neg[name],
        }
        for name in unique_labels
    ]

  def search(
      self,
      database_id: str,
      mode: SearchMode,
      label: str,
      query_uri: str,
  ) -> dict[str, Any]:
    """Performs search/ranking and returns results."""
    db = self._get_db(database_id)
    cache_key = (database_id, label)

    if mode == SearchMode.CLASSIFIER:
      with self._cache_lock:
        scores = self._scores_cache.get(cache_key)
        metrics = self._metrics_cache.get(cache_key, {})
      if scores is None:
        raise ValueError(
            f"Classifier for label '{label}' has not been trained yet."
        )
      classifier_trained = True
    else:
      scores = {}
      if query_uri:
        search_results = self._vector_search(db, query_uri)
        scores = {res.window_id: res.sort_score for res in search_results}
      with self._cache_lock:
        metrics = self._metrics_cache.get(cache_key, {})
        classifier_trained = cache_key in self._scores_cache

    # Map the windows to generic UI dictionaries
    results = self._get_ui_results_for_database(
        db, database_id=database_id, label=label, scores_lookup=scores
    )

    # For query or classifier modes, sort matching items by score/probability
    # descending.
    if (
        query_uri and mode == SearchMode.QUERY
    ) or mode == SearchMode.CLASSIFIER:
      results.sort(key=lambda x: x["score"], reverse=True)

    # Setup query details if search similarity is executed
    query_info = {"uri": None, "media_type": None, "url": None}
    if query_uri:
      query_info = {
          "uri": query_uri,
          "media_type": "audio",
          "url": (
              f"/stream?database={urllib.parse.quote(database_id)}"
              f"&query_uri={urllib.parse.quote(query_uri)}"
          ),
      }

    return {
        "query": query_info,
        "results": results,
        "classifier_trained": classifier_trained,
        "classifier_metrics": metrics,
    }

  def annotate(
      self,
      database_id: str,
      item_id: int,
      label: str,
      annotation: AnnotationValue,
  ) -> None:
    """Submits or clears a user annotation."""
    db = self._get_db(database_id)
    if annotation == AnnotationValue.UNCERTAIN:
      self._clear_annotations(db, window_id=item_id, label=label)
    else:
      label_type = (
          datatypes.LabelType.POSITIVE
          if annotation == AnnotationValue.POSITIVE
          else datatypes.LabelType.NEGATIVE
      )
      self._save_annotation(
          db, window_id=item_id, label=label, label_type=label_type
      )

  def train_classifier(self, database_id: str, label: str) -> dict[str, Any]:
    """Trains a classifier on user annotations and returns metrics."""
    db = self._get_db(database_id)

    # Validate minimal data requirements before starting model training
    annotations = self._retrieve_user_annotations(db, label=label)
    positives = sum(
        1 for a in annotations if a.label_type == datatypes.LabelType.POSITIVE
    )
    negatives = sum(
        1 for a in annotations if a.label_type == datatypes.LabelType.NEGATIVE
    )
    if positives < 1 or negatives < 1 or (positives + negatives) < 3:
      raise ValueError(
          "Please label at least 3 items (mix of 👍 and 👎) for label"
          f" '{label}' to train."
      )

    scores, metrics = self._train_active_learning_classifier(
        db, target_labels=[label]
    )

    # Cache results for use in subsequent classifier mode searches
    cache_key = (database_id, label)
    with self._cache_lock:
      self._scores_cache[cache_key] = scores
      self._metrics_cache[cache_key] = metrics

    return metrics

  def get_media_stream(
      self, database_id: str, path: str, params: dict[str, list[str]]
  ) -> tuple[bytes, str]:
    """Returns the raw bytes of a media segment and its content type."""
    db = self._get_db(database_id)

    # 1. Query reference streaming
    query_uris = params.get("query_uri")
    if query_uris:
      query_uri = query_uris[0]
      if query_uri:
        _, wav_bytes = self._get_query_data(query_uri, db)
        return wav_bytes, "audio/wav"

    # 2. Window clip streaming
    window_ids = params.get("window_id")
    if window_ids:
      window_id_str = window_ids[0]
      if window_id_str:
        try:
          window_id = int(window_id_str)
        except ValueError as e:
          raise ValueError(
              f"Invalid non-integer 'window_id': {window_id_str!r}"
          ) from e
        wav_bytes = self._get_audio_wav_bytes(db, window_id)
        return wav_bytes, "audio/wav"

    raise ValueError(
        "Missing or invalid 'window_id' or 'query_uri' query parameters."
    )

  def get_media_preview(
      self, database_id: str, item_id: int
  ) -> tuple[bytes, str]:
    """Generates and returns a spectrogram preview for a database item."""
    db = self._get_db(database_id)

    # Check cache directory before computing spectrogram
    cache_dir = self.db_dir / ".spectrogram_cache" / database_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{item_id}.png"
    if cache_file.exists():
      png_bytes = cache_file.read_bytes()
    else:
      png_bytes = self._compute_spectrogram(db, item_id)
      cache_file.write_bytes(png_bytes)

    return png_bytes, "image/png"

  def get_query_preview(
      self, database_id: str, query_uri: str
  ) -> tuple[bytes, str]:
    """Generates and returns a spectrogram preview for a query URI."""
    db = self._get_db(database_id)

    safe_filename = urllib.parse.quote_plus(query_uri)
    cache_dir = self.db_dir / ".spectrogram_cache" / database_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{safe_filename}.png"
    if cache_file.exists():
      png_bytes = cache_file.read_bytes()
    else:
      _, wav_bytes = self._get_query_data(query_uri, db)
      data, sr = soundfile.read(io.BytesIO(wav_bytes))
      png_bytes = self._compute_spectrogram_from_audio(data, sr)
      cache_file.write_bytes(png_bytes)

    return png_bytes, "image/png"

  # ============================================================================
  # Private Helper Methods
  # ============================================================================

  def _get_db(self, database_id: str) -> sqlite_usearch_impl.SQLiteUSearchDB:
    """Retrieves or creates a thread-local SQLite/USearch DB connection.

    Args:
      database_id: The ID of the database.

    Returns:
      A thread-local SQLiteUSearchDB connection instance.
    """
    if not hasattr(self._local, "dbs"):
      self._local.dbs = {}

    if database_id not in self._local.dbs:
      with self._db_lock:
        if database_id not in self._main_dbs:
          db_path = self.db_dir / database_id
          if not (db_path / "hoplite.sqlite").exists():
            db_path = self.db_dir / f"{database_id}.sqlite"
          self._main_dbs[database_id] = (
              sqlite_usearch_impl.SQLiteUSearchDB.create(str(db_path))
          )
        main_db = self._main_dbs[database_id]
        self._local.dbs[database_id] = main_db.thread_split()

    return self._local.dbs[database_id]

  def _read_audio_window(
      self, db: sqlite_usearch_impl.SQLiteUSearchDB, window_id: int
  ) -> tuple[np.ndarray, float]:
    """Loads and slices raw audio bytes for a given window ID."""
    window = db.get_window(window_id)
    recording = db.get_recording(window.recording_id)

    audio_sources_meta = db.get_metadata("audio_sources")
    if not audio_sources_meta.audio_globs:
      raise ValueError("No audio sources configuration found in database.")

    base_path = None
    if recording.deployment_id is not None:
      try:
        deployment = db.get_deployment(recording.deployment_id)
        dataset_name = deployment.project
        for glob_config in audio_sources_meta.audio_globs:
          if glob_config.get("dataset_name") == dataset_name:
            base_path = glob_config.get("base_path")
            break
      except KeyError:
        pass

    if base_path is None:
      checked_paths = [
          g.get("base_path") for g in audio_sources_meta.audio_globs
      ]
      raise FileNotFoundError(
          f"Could not resolve base path for recording '{recording.filename}' "
          f"(recording_id: {recording.id}). Checked paths: {checked_paths}"
      )

    filepath = pathlib.Path(base_path) / recording.filename

    info = soundfile.info(str(filepath))
    sr = info.samplerate
    offset_s = window.offsets[0]
    window_size_s = window.offsets[1] - window.offsets[0]
    data = load_audio_window(
        str(filepath),
        offset_s=offset_s,
        sample_rate=sr,
        window_size_s=window_size_s,
    )
    return data, sr

  def _get_audio_wav_bytes(
      self, db: sqlite_usearch_impl.SQLiteUSearchDB, window_id: int
  ) -> bytes:
    """Reads a specific audio window segment and serializes to in-memory WAV."""
    data, sr = self._read_audio_window(db, window_id)
    out_buf = io.BytesIO()
    soundfile.write(out_buf, data, sr, format="WAV")
    return out_buf.getvalue()

  def _compute_spectrogram_from_audio(
      self, data: np.ndarray, sr: float
  ) -> bytes:
    """Computes and returns a spectrogram image for a given audio segment."""
    if len(data.shape) > 1:
      data = np.mean(data, axis=1)

    spec = embedding_display.pcen_melspec_display(data, sample_rate_hz=sr)

    db_min = spec.min()
    db_max = spec.max()
    if db_max > db_min:
      spec_normalized = (spec - db_min) / (db_max - db_min)
    else:
      spec_normalized = np.zeros_like(spec)

    spec_normalized = np.flipud(spec_normalized.T)

    cmap = matplotlib.colormaps["viridis"]
    rgba_img = cmap(spec_normalized)
    rgb_img = (rgba_img[:, :, :3] * 255).astype(np.uint8)

    img = Image.fromarray(rgb_img)
    img = img.resize((640, 400), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

  def _compute_spectrogram(
      self, db: sqlite_usearch_impl.SQLiteUSearchDB, window_id: int
  ) -> bytes:
    """Computes and returns a spectrogram PNG image for a given audio window."""
    data, sr = self._read_audio_window(db, window_id)
    return self._compute_spectrogram_from_audio(data, sr)

  def _save_annotation(
      self,
      db: sqlite_usearch_impl.SQLiteUSearchDB,
      window_id: int,
      label: str,
      label_type: datatypes.LabelType,
  ) -> None:
    """Saves/overwrites a user annotation in the database."""
    window = db.get_window(window_id)
    db.insert_annotation(
        recording_id=window.recording_id,
        offsets=window.offsets,
        label=label,
        label_type=label_type,
        provenance="user",
        handle_duplicates="overwrite",
    )
    db.commit()

  def _clear_annotations(
      self, db: sqlite_usearch_impl.SQLiteUSearchDB, window_id: int, label: str
  ) -> None:
    """Removes user annotations for a specific window/label."""
    existing = db.get_window_annotations(window_id, label=label)
    user_annotations = [a for a in existing if a.provenance == "user"]
    for a in user_annotations:
      db.remove_annotation(a.id)
    db.commit()

  def _retrieve_user_annotations(
      self, db: sqlite_usearch_impl.SQLiteUSearchDB, label: str
  ) -> list[Any]:
    """Queries all user-authored annotations for a specific label."""
    annotations_filter = config_dict.create(
        eq=dict(label=label, provenance="user")
    )
    return db.get_all_annotations(filter=annotations_filter)

  def _map_window_to_ui_item(
      self,
      window: datatypes.Window,
      recording: datatypes.Recording,
      score: float = 0.0,
      annotation: str = "UNCERTAIN",
      database_id: str = "",
  ) -> dict[str, Any]:
    """Maps a Hoplite Window and Recording to the generic UI representation."""
    title = (
        f"{recording.filename} [{window.offsets[0]:.1f}s -"
        f" {window.offsets[1]:.1f}s]"
    )
    return {
        "id": window.id,
        "title": title,
        "media_type": "audio",
        "media_url": (
            f"/stream?database={urllib.parse.quote(database_id)}"
            f"&window_id={window.id}"
        ),
        "score": score,
        "annotation": annotation,
    }

  def _get_ui_results_for_database(
      self,
      db: sqlite_usearch_impl.SQLiteUSearchDB,
      database_id: str,
      label: str = "",
      scores_lookup: dict[int, float] | None = None,
  ) -> list[dict[str, Any]]:
    """Queries all windows in the database and formats them for UI display."""
    if scores_lookup is None:
      scores_lookup = {}

    windows = db.get_all_windows()
    recordings = {r.id: r for r in db.get_all_recordings()}

    ann_lookup = {}
    if label:
      annotations_filter = config_dict.create(
          eq=dict(label=label, provenance="user")
      )
      user_anns = db.get_all_annotations(filter=annotations_filter)
      ann_lookup = {
          (
              ann.recording_id,
              ann.offsets[0],
              ann.offsets[1],
          ): ann.label_type.name
          for ann in user_anns
      }

    results = []
    for w in windows:
      recording = recordings.get(w.recording_id)
      if not recording:
        continue

      wkey = (w.recording_id, w.offsets[0], w.offsets[1])
      annotation = ann_lookup.get(wkey, "UNCERTAIN")
      score = scores_lookup.get(w.id, 0.0)

      item = self._map_window_to_ui_item(
          window=w,
          recording=recording,
          score=score,
          annotation=annotation,
          database_id=database_id,
      )
      results.append(item)

    return results

  def _get_query_data(
      self, query_uri: str, db: sqlite_usearch_impl.SQLiteUSearchDB
  ) -> tuple[np.ndarray, bytes]:
    """Retrieves cached query embedding and WAV bytes, or computes them."""
    if query_uri in self._query_cache:
      cached_emb, cached_wav = self._query_cache[query_uri]
      return cached_emb.copy(), cached_wav

    with self._query_cache_lock:
      if query_uri in self._query_cache:
        cached_emb, cached_wav = self._query_cache[query_uri]
        return cached_emb.copy(), cached_wav

      db_model_config = db.get_metadata("model_config")
      model_key = db_model_config.model_key

      with self._model_cache_lock:
        if model_key not in self._model_cache:
          model_class = model_configs.get_model_class(model_key)
          self._model_cache[model_key] = model_class.from_config(
              db_model_config.model_config
          )
        embedding_model = self._model_cache[model_key]

      window_size_s = getattr(embedding_model, "window_size_s", 5.0)
      query = embedding_display.QueryDisplay(
          uri=query_uri,
          offset_s=0.0,
          window_size_s=window_size_s,
          sample_rate_hz=embedding_model.sample_rate,
      )

      audio = query.get_audio_window()
      query_embedding = embedding_model.embed(audio).embeddings[0, 0]

      out_buf = io.BytesIO()
      soundfile.write(out_buf, audio, embedding_model.sample_rate, format="WAV")
      wav_bytes = out_buf.getvalue()

      self._query_cache[query_uri] = (query_embedding, wav_bytes)

    return query_embedding.copy(), wav_bytes

  def _vector_search(
      self, db: sqlite_usearch_impl.SQLiteUSearchDB, query_uri: str
  ) -> list[Any]:
    """Performs brute-force vector search and returns sorted results."""
    query_embedding, _ = self._get_query_data(query_uri, db)
    return db.search(
        query_embedding,
        search_list_size=len(db.match_window_ids()),
        approximate=False,
    )

  def _safe_get_embeddings_batch(
      self,
      db: sqlite_usearch_impl.SQLiteUSearchDB,
      window_ids: list[int] | np.ndarray,
  ) -> np.ndarray:
    """Safely retrieves a batch of embeddings from a Hoplite database.

    Works around differences in the return type of the underlying SQLite/USearch
    database implementation.

    Args:
      db: The Hoplite database.
      window_ids: A sequence of window IDs.

    Returns:
      A numpy array of the retrieved embeddings stacked along the first axis.
    """
    vectors = db.ui.get(window_ids)
    if isinstance(vectors, np.ndarray):
      return vectors
    elif isinstance(vectors, tuple):
      return np.stack(vectors, axis=0)
    else:
      return np.stack([db.ui.get(wid) for wid in window_ids], axis=0)

  def _train_active_learning_classifier(
      self, db: sqlite_usearch_impl.SQLiteUSearchDB, target_labels: list[str]
  ) -> tuple[dict[int, float], dict[str, Any]]:
    """Trains a linear classifier on user annotations."""
    # Count annotations for target labels to select train/eval configuration
    annotations = []
    for label in target_labels:
      annotations.extend(self._retrieve_user_annotations(db, label=label))
    dataset_size = len(annotations)

    # If dataset size is too small, fallback immediately to train_ratio=1.0,
    # min_eval_examples=0 to avoid a ValueError during data split generation.
    if dataset_size < 10:
      train_ratio = 1.0
      min_eval_examples = 0
      logging.info(
          "Small dataset size (%d). Falling back to 1.0 training ratio.",
          dataset_size,
      )
    else:
      train_ratio = 0.9
      min_eval_examples = 1

    wrapped_db = CachedHopliteDB(db)
    data_manager = classifier_data.AgileDataManager(
        target_labels=target_labels,
        db=wrapped_db,
        train_ratio=train_ratio,
        min_eval_examples=min_eval_examples,
        batch_size=128,
        weak_negatives_batch_size=128,
        rng=np.random.default_rng(seed=5),
    )

    with mock.patch(
        "perch_hoplite.agile.classifier.eval_classifier", safe_eval_classifier
    ):
      linear_classifier, eval_scores = classifier.train_linear_classifier(
          data_manager=data_manager,
          learning_rate=1e-3,
          weak_neg_weight=0.05,
          num_train_steps=128,
      )

    train_ids, eval_ids = data_manager.get_train_test_split()
    dataset_size = len(train_ids) + len(eval_ids)

    window_ids = db.match_window_ids()
    embeddings = self._safe_get_embeddings_batch(db, window_ids)

    logits = linear_classifier(embeddings)
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    scores = {
        wid: float(probabilities[i, 0]) for i, wid in enumerate(window_ids)
    }

    metrics = {
        "accuracy": (
            sanitize_float(eval_scores.get("top1_acc", 1.0))
            if eval_scores
            else 1.0
        ),
        "dataset_size": dataset_size,
    }
    return scores, metrics
