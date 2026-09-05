# Agentic Ecology Workspace Guidelines

## Environment & Workspace Management

- **Environment Management:** This repository is managed exclusively with
  `uv`. All dependency resolution and environment setup are handled by `uv`.
  The `soundfile` library is guaranteed to be available in this environment;
  do not waste steps verifying its installation.
- **Agent Artifacts:** You MUST store artifacts (Python scripts, Markdown
  documents, etc.) in the `agent_workspace` directory inside the repository's
  root directory so that the user can inspect your work. Do NOT create or
  commit any temporary, throwaway, or scratch files anywhere else.
- **Code Execution:** The correct and mandatory way to interact with Python
  scripts or modules is via the `uv run python` command prefix. This ensures
  the correct environment and dependencies (such as `perch-hoplite`) are used.
  Any code you run MUST be in the form of a script in the `agent_workspace`
  directory.
- **Adherence to Skills & Templates:** You MUST closely review and adhere to
  the design patterns, utility functions, and thread-safety patterns defined
  in the skill reference templates. When adapting reference templates into
  your project workspace (e.g., `agent_workspace/`), ensure all
  template-provided safety guards are fully preserved and applied to live data
  flows to prevent known edge-case failures (such as NaN serialization or
  multi-threaded SQLite connection errors).

## Compute Assessment & Execution Planning

When faced with tasks requiring significant compute (e.g., generating embeddings
across large audio/image datasets):

- **Evaluate Execution Options:** Do not unilaterally launch long-running or
  resource-intensive jobs on the local machine without evaluation. Reflect on
  the available options:
  - **Local Execution (`uv run python agent_workspace/...`):**
    - *Pros:* Zero remote setup or cloud dependency; no cloud compute unit
      / quota usage; outputs and databases remain directly in the local
      workspace.
    - *Cons:* Constrained by local machine hardware (often CPU-only or
      limited memory/VRAM); can take significant wall-clock time and
      throttle the local system.
  - **Remote Execution via Colab (`colab-operator` / `colab` CLI):**
    - *Pros:* Access to high-throughput GPU/TPU accelerators (T4, L4,
      A100, TPU v5e/v6e); drastically reduces embedding/training
      wall-clock time; frees local compute.
    - *Cons:* Consumes Google Colab compute units; requires
      authentication, remote package setup, and dataset transfer / result
      syncing.
  - **Cloud-Scale Execution via GCP Dataflow (`DataflowRunner` & Cloud Storage):**
    - *Pros:* Massive distributed horizontal scaling across worker pools;
      efficiently processes large-scale corpora (hundreds of GBs to TBs of
      PAM recordings or camera trap imagery) that exceed single-machine or
      Colab session limits; outputs durable, sharded embeddings directly on
      Google Cloud Storage.
    - *Cons:* Incurs Google Cloud infrastructure costs; requires GCP
      project authentication, IAM permissions, and API enablement
      (`dataflow.googleapis.com`, `storage.googleapis.com`).
- **Present Recommendation & Await Decision:**
  - Weigh dataset volume, estimated runtime, local hardware capabilities,
    and setup overhead.
  - Present the options, key trade-offs, and a recommended approach clearly
    to the user (e.g. Local for small batches, Colab for intermediate
    accelerator needs, GCP Dataflow for massive dataset corpora).
  - Prompt the user for their preference and proceed only after the user
    chooses how to run the job.
- **Execute Chosen Path:**
  - If **Local**: Follow local execution standards using `uv run python`.
  - If **Colab**: Use the `colab-operator` skill (e.g., ephemeral `colab run`) and storage workflows as appropriate.
    - **Files within Upload Limit (< ~70MB):** Transfer directly using
      `colab upload`.
    - **Files Exceeding Upload Limit (≥ ~70MB):** `colab upload` enforces
      a 100MB HTTP request payload limit, meaning raw files exceeding
      ~70MB fail due to base64 encoding expansion. When transferring files
      or datasets that exceed this threshold, do not unilaterally choose
      a transfer strategy. Follow the evaluate/recommend/await decision
      protocol:
      - **Evaluate Storage Options:**
        - **Chunking & Reassembly:** Split files locally (e.g.,
          `split -b 50M`), upload chunks via `colab upload`, and
          reconstruct remotely (`cat ...`).
          - *Pros:* Self-contained in Colab VM; requires no
            Google Drive OAuth permissions or interactive mount
            prompts.
          - *Cons:* Overhead to split and reassemble; data is
            ephemeral and lost if the runtime disconnects.
        - **Google Drive (`gws-drive-upload` + Drive Mount):** Upload
          to Drive via `gws-drive-upload` and mount inside Colab
          (`drive.mount('/content/drive')`).
          - *Pros:* Robust for large multi-GB files; persistent
            across runtimes and sessions.
          - *Cons:* Requires Google Drive OAuth permissions and
            session mounting steps.
      - **Present Storage Recommendation & Await Decision:**
        - Weigh dataset size, persistence requirements, and setup
          friction.
        - Present the storage options, key trade-offs, and a
          recommended approach clearly to the user.
        - Prompt the user for their preference and proceed only after
          the user decides how to handle file storage.
      - **Execute Chosen Storage Path:** Proceed with the selected
        strategy (chunking or Google Drive upload and mount).
  - If **GCP (Dataflow & Cloud Storage)**:
    - **Use Installed Google Skills:** Strictly use the official
      `google/skills` skills (`gcloud`, `google-cloud-storage-basics`,
      `google-cloud-storage-bucket-architect`, `google-cloud-storage-fuse`,
      `cloud-logging-query-generation`) for all GCP operations.
    - **Non-Interactive Execution:** Always supply `--quiet` (or `-q`) to
      `gcloud` commands and explicitly pass `--project=<PROJECT_ID>` and
      regional flags (e.g., `--region=<REGION>`) to prevent interactive
      prompts from hanging execution.
    - **Storage Architecture & Staging:** Use
      `google-cloud-storage-basics` (`gcloud storage cp` or `rsync -r`)
      for multi-threaded asset staging into standard bucket layouts
      (`gs://<BUCKET>/audio/...`, `gs://<BUCKET>/embeddings/...`,
      `gs://<BUCKET>/temp/`, `gs://<BUCKET>/staging/`).
    - **Pipeline Execution:** Launch Beam jobs with `--runner=DataflowRunner`
      using the relevant domain skill template (e.g. `dataflow_embed.py` in
      bioacoustics).
    - **Troubleshooting & Logging:** Use `cloud-logging-query-generation`
      to inspect Dataflow worker logs and diagnose failed stages or OOMs.

## Technical Gotchas & Rules

### 1. macOS Dynamic Library Deadlock (TensorFlow & PyArrow)

On macOS, both `tensorflow` and `pyarrow` (Apache Arrow) statically link Abseil
(`absl`) but expose their symbols globally. Due to macOS's namespace resolution,
if `pyarrow` is loaded first, `tensorflow` will bind to incompatible Abseil
symbols, causing a compiler deadlock.

- **Rule**: Always force `import tensorflow as tf` at the absolute top of any
  Python script or entry point that uses JAX or TensorFlow.
- **Rule**: Ensure this import precedes any imports of `perch_hoplite`,
  `pandas`, `gcsfs`, `fsspec`, or packages that transitively load `pyarrow`.

### 1b. Linux PyTorch & TensorFlow Import Conflict (Segmentation Fault)

On Linux, there is a symbol conflict between PyTorch (`yolov5`) and TensorFlow.
If `tensorflow` is imported first, subsequent imports of `yolov5` will segfault.

- **Rule**: If a script imports both `yolov5` and `tensorflow`, `import yolov5` MUST be placed at the absolute top of the script, preceding `import tensorflow as tf`.

### 2. Suppressing Internal SQL Trace Logs

The `perch-hoplite` database adapter logs every executed query at `INFO` level,
clogging logs.

- **Rule**: Implement a targeted `logging.Filter` to discard only the
  `"Executed SQL statement"` entries at your script's entry point:

  ```python
  import logging
  class SQLSuppressFilter(logging.Filter):
    def filter(self, record):
      return "Executed SQL statement" not in record.getMessage()
  logging.getLogger("absl").addFilter(SQLSuppressFilter())
  ```
