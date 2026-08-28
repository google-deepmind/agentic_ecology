# Agentic Ecology Workspace Guidelines

## Environment & Workspace Management

*   **Environment Management:** This repository is managed exclusively with
    `uv`. All dependency resolution and environment setup are handled by `uv`.
    The `soundfile` library is guaranteed to be available in this environment;
    do not waste steps verifying its installation.
*   **Agent Artifacts:** You MUST store artifacts (Python scripts, Markdown
    documents, etc.) in the `agent_workspace` directory inside the repository's
    root directory so that the user can inspect your work. Do NOT create or
    commit any temporary, throwaway, or scratch files anywhere else.
*   **Code Execution:** The correct and mandatory way to interact with Python
    scripts or modules is via the `uv run python` command prefix. This ensures
    the correct environment and dependencies (such as `perch-hoplite`) are used.
    Any code you run MUST be in the form of a script in the `agent_workspace`
    directory.
*   **Adherence to Skills & Templates:** You MUST closely review and adhere to
    the design patterns, utility functions, and thread-safety patterns defined
    in the skill reference templates. When adapting reference templates into
    your project workspace (e.g., `agent_workspace/`), ensure all
    template-provided safety guards are fully preserved and applied to live data
    flows to prevent known edge-case failures (such as NaN serialization or
    multi-threaded SQLite connection errors).

## Compute Assessment & Execution Planning

When faced with tasks requiring significant compute (e.g., generating embeddings
across large audio/image datasets):

*   **Evaluate Execution Options:** Do not unilaterally launch long-running or
    resource-intensive jobs on the local machine without evaluation. Reflect on
    the available options:
    *   **Local Execution (`uv run python agent_workspace/...`):**
        *   *Pros:* Zero remote setup or cloud dependency; no cloud compute unit
            / quota usage; outputs and databases remain directly in the local
            workspace.
        *   *Cons:* Constrained by local machine hardware (often CPU-only or
            limited memory/VRAM); can take significant wall-clock time and
            throttle the local system.
    *   **Remote Execution via Colab (`colab-operator` / `colab` CLI):**
        *   *Pros:* Access to high-throughput GPU/TPU accelerators (T4, L4,
            A100, TPU v5e/v6e); drastically reduces embedding/training
            wall-clock time; frees local compute.
        *   *Cons:* Consumes Google Colab compute units; requires
            authentication, remote package setup, and dataset transfer / result
            syncing (e.g., via Google Drive / `rclone`).
*   **Present Recommendation & Await Decision:**
    *   Weigh dataset volume, estimated runtime, local hardware capabilities,
        and setup overhead.
    *   Present the options, key trade-offs, and a recommended approach clearly
        to the user.
    *   Prompt the user for their preference and proceed only after the user
        chooses how to run the job.
*   **Execute Chosen Path:**
    *   If **Local**: Follow local execution standards using `uv run python`.
    *   If **Colab**: Use the `colab-operator` skill (e.g., ephemeral `colab
        run`) and storage workflows as appropriate.

## Technical Gotchas & Rules

### 1. macOS Dynamic Library Deadlock (TensorFlow & PyArrow)

On macOS, both `tensorflow` and `pyarrow` (Apache Arrow) statically link Abseil
(`absl`) but expose their symbols globally. Due to macOS's namespace resolution,
if `pyarrow` is loaded first, `tensorflow` will bind to incompatible Abseil
symbols, causing a compiler deadlock.

*   **Rule**: Always force `import tensorflow as tf` at the absolute top of any
    Python script or entry point that uses JAX or TensorFlow.
*   **Rule**: Ensure this import precedes any imports of `perch_hoplite`,
    `pandas`, `gcsfs`, `fsspec`, or packages that transitively load `pyarrow`.

### 1b. Linux PyTorch & TensorFlow Import Conflict (Segmentation Fault)

On Linux, there is a symbol conflict between PyTorch (`yolov5`) and TensorFlow.
If `tensorflow` is imported first, subsequent imports of `yolov5` will segfault.

*   **Rule**: If a script imports both `yolov5` and `tensorflow`, `import
    yolov5` MUST be placed at the absolute top of the script, preceding `import
    tensorflow as tf`.

### 2. Suppressing Internal SQL Trace Logs

The `perch-hoplite` database adapter logs every executed query at `INFO` level,
clogging logs.

*   **Rule**: Implement a targeted `logging.Filter` to discard only the
    `"Executed SQL statement"` entries at your script's entry point:

    ```python
    import logging
    class SQLSuppressFilter(logging.Filter):
      def filter(self, record):
        return "Executed SQL statement" not in record.getMessage()
    logging.getLogger("absl").addFilter(SQLSuppressFilter())
    ```
