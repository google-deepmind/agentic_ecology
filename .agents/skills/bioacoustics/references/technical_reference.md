# Bioacoustics Technical Reference

Detailed instructions and code examples for performing bioacoustic analysis
using the `perch-hoplite` Python package.

## Preprocessing

### Selecting the Embedding Model

*   **Valid Options**: Valid options are defined in
    `perch_hoplite.zoo.model_configs.ModelConfigName`'s enum values.
*   **User Interaction**: When presenting options to the user, always present
    **all** of those values. Verify the user's choice and flag/warn if the
    specified name does not correspond to any valid preset.

### Creating and Populating the Hoplite Database

Refer to the `create_and_populate_db` function in [server.py](server.py).

*   Favor `perch_hoplite.db.sqlite_usearch_impl.SQLiteUSearchDB.create(db_path)`
    to connect to an existing database, as it automatically loads the
    configuration from the database. Note that `db_path` must be the directory
    containing the database files (e.g. `databases/powdermill`), not the path to
    the sqlite file itself.
*   Favor `perch_hoplite.agile.embed.EmbedWorker(audio_sources, model_config,
    db)` to populate the database. This high-level API automatically manages
    dataset configurations, creates deployments and recordings, processes audio
    (including optional sharding), generates embeddings, and saves essential
    metadata (`model_config` and `audio_sources`) to the database.
*   **Recursive Directory Globbing Limitation:** The `file_glob` parameter of
    `AudioSourceConfig` does **not** support recursive globbing patterns such as
    `**/*.wav`.
*   Unless the user provides an explicit instruction to the contrary, use
    `min_audio_len_s=1.0` and `target_sample_rate_hz=-2` (i.e., the model's
    target sample rate).

### Hoplite Data Schema & API

When querying the database, you will interact with the following key datatypes
(defined in `perch_hoplite.db.datatypes`):

*   **Window**: Represents a segment of audio.
    *   `id`: `int` (the unique window ID, also used as key in the USearch
        index).
    *   `recording_id`: `int` (links to the corresponding `Recording`).
    *   `offsets`: `list[float]` (start and end times in seconds, e.g., `[0.0,
        5.0]`).
    *   `embedding`: `np.ndarray` (the embedding vector, if loaded).
*   **Recording**: Represents a full audio file.
    *   `id`: `int`.
    *   `filename`: `str` (path relative to the dataset's `base_path`).
    *   `deployment_id`: `int` (links to `Deployment`).
*   **Deployment**: Represents a site deployment.
    *   `id`: `int`.
    *   `name`: `str`.
    *   `project`: `str`.
*   **Database Metadata**: Refer to the `inspect_database` function in
    [server.py](server.py). Extract model config, key (e.g.
    `taxonomy_model_tf`), and database window counts.
*   **Database Adapter**: Refer to [server.py](server.py) for converting
    hierarchical database structures into the flat JSON representation expected
    by the frontend UI.

#### `SQLiteUSearchDB` Key Methods

To query and inspect the database, use the following methods on the
`SQLiteUSearchDB` object:

*   `db.get_all_labels() -> list[str]`: Retrieves all distinct label names
    present in the database's annotations.
*   `db.get_all_recordings() -> list[Recording]`: Retrieves all recordings
    present in the database.
*   `db.get_all_windows() -> list[Window]`: Retrieves all audio windows present
    in the database.
*   `db.get_window(window_id: int) -> Window`: Retrieves a specific window by
    its ID.
*   `db.get_recording(recording_id: int) -> Recording`: Retrieves a specific
    recording by its ID.
*   `db.get_metadata(key: str) -> Any`: Retrieves deserialized metadata saved in
    the metadata table (e.g., `'model_config'` or `'audio_sources'`).
*   `db.match_window_ids() -> list[int]`: Returns a list of all window IDs in
    the database.

--------------------------------------------------------------------------------

## Bioacoustics Web App

### Backend

Follow the best practices laid out in this repository's `ui` skill.

#### SQLite Concurrency & Thread-Splitting

Refer to the `_get_db` implementation in [server.py](server.py).

SQLite connections are bound to individual threads and cannot be shared. In a
multi-threaded Python server (e.g., using `socketserver.ThreadingTCPServer`),
you must use the Perch Hoplite `db.thread_split()` method to spawn isolated
database connections for each handler thread.

#### Resolving Audio Files

To locate the physical audio file on disk for a `Recording` object:

1.  Retrieve `audio_sources` metadata: `audio_sources_meta =
    db.get_metadata("audio_sources")`.
2.  Retrieve the matching deployment to find the corresponding project name:
    `deployment = db.get_deployment(recording.deployment_id)` and
    `dataset_name = deployment.project`.
3.  Match the `dataset_name` to the correct `AudioSourceConfig` in
    `audio_sources_meta.audio_globs` to find its `base_path`. If not found,
    raise a `FileNotFoundError`. *(Note: Use key-based access rather than
    attribute access for `audio_globs` elements as they are standard dicts).*
4.  Combine with the recording's filename: `os.path.join(base_path,
    recording.filename)`.

#### Streaming/Reading Audio Segments

Refer to the `_get_audio_wav_bytes` function in [server.py](server.py). Read a
specific window's audio segment and convert to WAV bytes in-memory using
`soundfile` and `io.BytesIO`.

For examples of integrating this streaming logic within a multi-threaded web
server handler (including connection management and error handling), refer to
`HopliteDatabaseAdapter.get_media_stream` in [server.py](server.py).

#### Generating Spectrogram Previews

Refer to the `_compute_spectrogram` function in [server.py](server.py). To
generate high-quality visual spectrogram previews aligned with the model's
visual representation:

1.  Load the raw numpy slice for the audio window.
2.  Compute the PCEN mel-spectrogram matrix using
    `perch_hoplite.agile.embedding_display.pcen_melspec_display(data,
    sample_rate_hz=sr)`.
3.  Normalize the resulting spectrogram values to `[0, 1]`.
4.  Flip the transposed spectrogram matrix vertically: `np.flipud(spec.T)`.
5.  Map the normalized matrix values through a colormap (e.g.
    `matplotlib.colormaps['viridis']`), convert to 8-bit RGB, resize the image
    array to standard UI card dimensions (e.g. `320x100`), and output as PNG
    bytes.

**Caching Recommendation**: Since calculating spectrograms is CPU-intensive,
cache the generated PNG bytes locally (e.g., in a directory like
`agent_workspace/.spectrogram_cache`) keying by the database and item ID. Check
this cache before computing.

#### Annotations

*   **Retrieving Label-Specific Annotations from the Database:** When the label
    changes in the UI, retrieve all annotations across the database for the new
    label value. Refer to the `_retrieve_user_annotations` function in
    [server.py](server.py).
*   **Saving Annotations:** Insert/overwrite annotation when labeled. Use a
    distinct tag (e.g., `"user"`) to distinguish from programmatic labels. Refer
    to the `_save_annotation` function in [server.py](server.py).
*   **Clearing Annotations:** Remove existing `"user"` annotations if reset to
    unsure. Refer to the `_clear_annotations` function in
    [server.py](server.py).
*   **Restoring State**: Fetch existing annotations on page load (`/api/search`)
    to highlight existing annotations. Refer to the
    `_get_ui_results_for_database` function in [server.py](server.py).

#### Vector Search

Refer to the `_vector_search` function in [server.py](server.py).

*   **SearchResult Schema**: The results yielded by iterating over
    `db.search(...)` (which returns a `TopKSearchResults` container) are
    `SearchResult` objects containing:
    *   `window_id`: `int` (corresponds to the window ID).
    *   `sort_score`: `float` (the similarity score, higher is better).
    *   *(Note: Historical references might use `result.key` and
        `result.distance`, but you must use `result.window_id` and
        `result.sort_score`)*.
*   **Xeno-Canto & Audio Loading**: The utility function `load_audio(path,
    target_sample_rate)` (imported from `perch_hoplite.audio_io`) is the
    recommended way to load query audio. It automatically handles both local
    file paths (relative to workspace or absolute) and remote Xeno-Canto IDs
    (e.g., `'xc105133'`), returning a 1D NumPy float array of audio samples.
*   **Query/Label Separation**: Ensure that the vector search query parameter
    (e.g., `query_uri`) is separated from the target annotation species class
    label (e.g., `label`). When running vector search similarity searches, rank
    the results by similarity to `query_uri` but display and save annotations
    under the name of the active `label` (e.g., `wood_thrush`). Do NOT save
    annotations using the query URI itself as the label.
*   **Caching Requirement**: Downloading is slow. When building a dynamic web
    server, **call `query.get_audio_window()` or `audio_io.load_audio` once,
    cache the NumPy array in-memory, and stream it as WAV from memory** for the
    `/query` endpoint. Do not trigger a fresh download on every request. Refer
    to the `_get_query_data` function in [server.py](server.py) for a
    recommended thread-safe query caching implementation.

#### Trained Classifier Search

Refer to the `_train_active_learning_classifier` function in
[server.py](server.py).

*   **Retrieving Split IDs & Dataset Size**: Call
    `data_manager.get_train_test_split()` (which returns a tuple of `(train_ids,
    eval_ids)`) to inspect splits and count total annotated examples. Do **not**
    access attributes like `train_idx` or `eval_idx` directly.
*   **Caching Requirement**: Training is slow. Unless the user specifically
    requests the classifier for a particular label to be retrained, rely on
    caching the trained classifier and use the cached classifier to rank
    database rows.
*   **USearch batch retrieval compatibility**: Depending on the version of the
    `usearch` index library installed, `db.get_embeddings_batch(window_ids)`
    (which internally calls `db.ui.get()`) may return a `tuple` of 1D arrays
    instead of a single 2D NumPy array, triggering a `RuntimeError`. To prevent
    this, implement a robust wrapper to fetch batches of embeddings. Refer to
    the `_safe_get_embeddings_batch` method in [server.py](server.py).

*   **AgileDataManager Split ValueError**: Refer to the
    `_train_active_learning_classifier` function in [server.py](server.py). When
    training on very small dataset sizes (e.g. 3 or 4 annotations),
    instantiating `AgileDataManager` with `train_ratio=0.9` and
    `min_eval_examples=1` raises a `ValueError` because the evaluation set
    cannot be populated. Use a try/except fallback block to train with
    `train_ratio=1.0` and `min_eval_examples=0`.

*   **Evaluation Metrics (`eval_scores`)**: Note that `train_linear_classifier`
    returns `eval_scores` as a dictionary, not a list. To extract the accuracy
    metric, look up `'top1_acc'` (e.g., `eval_scores.get('top1_acc', 1.0)`). Do
    NOT use list index lookup like `eval_scores[-1]`, which will raise a
    `KeyError: -1`.

    *   **NaN Alert**: When the evaluation set is empty (e.g., when using the
        small dataset fallback), `'top1_acc'` will be `NaN`. When exposing this
        metric in a web API, you MUST sanitize it (e.g. using `sanitize_float`)
        to prevent JSON serialization errors in the browser.

### Frontend

Follow the best practices laid out in this repository's `ui` skill.

#### Bioacoustic Specificities

*   **Media Preview**: Adapt the UI card to present a visual spectrogram image
    loaded from the `/api/preview` endpoint (which maps to
    `compute_spectrogram`). Implement play-on-hover logic with progress
    tracking: when the cursor hovers over the spectrogram, immediately query and
    trigger playback on the corresponding `<audio>` element; when the cursor
    leaves, pause, rewind, and reset progress. Sync a vertical overlay line
    (`#progress-bar-${itemId}`) with the audio's `timeupdate` event to provide a
    real-time sliding progress indicator.
*   **Dynamic Label Integration**: Retrieve the label dynamically from query
    parameters (`label` in `/api/search` and `/api/train`) or request bodies
    (`label` in `/api/annotation`) rather than hardcoding, ensuring the user can
    control the active database label via the UI's text input field.
*   **Active Training**: Map the `/api/train?label=...` POST endpoint to trigger
    `classifier.train_linear_classifier` using SQLite annotations matching that
    specific `label`, and return accuracy metrics.
    *   This should only be exposed to the user if there are enough annotations
        for the label specified in the UI.
*   **Classifier Scoring**: Map the `/api/search?mode=classifier&label=...`
    endpoint to score database windows using the trained classifier, returning
    unlabeled items ranked by probability and populating the annotation field
    according to the active `label`.
*   **Annotation Integration**: Map the `/api/annotation` POST endpoint to
    insert or remove annotations directly in SQLite using the requested `label`
    instead of a hardcoded query URI.
