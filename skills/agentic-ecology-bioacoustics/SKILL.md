---
name: agentic-ecology-bioacoustics
description: >-
  Provides bioacoustic analysis capabilities for ecologists and researchers
  using the perch-hoplite Python package. A typical use case is to use agile
  modeling to bootstrap the creation and deployment of a bespoke detector for
  targeted species on an existing collection of passive acoustic monitoring
  recordings. Use when processing and annotating audio recordings for
  bioacoustic applications.
license: Apache-2.0
compatibility: Requires Python 3.12+, uv, TensorFlow, libsndfile
---

# Bioacoustics Skill

> [!CAUTION] **DATABASE SAFETY AND INTEGRITY**: Do NOT write, modify, or insert
> test annotations directly into the user's production databases. If you need to
> test database operations (such as saving annotations or training models), you
> MUST copy the database to a temporary location in `agent_workspace` (e.g.,
> `agent_workspace/test_db`) and test against the copy. Never leave testing data
> in production databases.

## Workflow Overview

Follow these sequential steps:

1. **Preprocess:** Create and populate a Hoplite database from the user's
   recordings.
   1. **Identify Recordings:** Locate the user's recordings. If the location
      is not provided, ask the user.
   1. **Select Embedding Model:** Confirm which embedding model to use (e.g.,
      `perch_v2`, `surfperch`). Query the user if they have not specified one.
   1. **Assess Compute Scale:** Follow the compute assessment and execution
      planning protocol in `AGENTS.md` to evaluate options (Local vs. Colab
      vs. GCP Dataflow) and await the user's decision.
   1. **Execute Ingestion Strategy:**
      > [!IMPORTANT] **MANDATORY INGESTION PIPELINE**: Always use [`dataflow_embed.py`](assets/dataflow_embed.py) to extract audio embeddings into sharded Apache Parquet files, followed by [`ingest_embeddings.py`](assets/ingest_embeddings.py) to build the Hoplite database. Do NOT use `EmbedWorker` directly.
      - **Option A (Local Ingestion via DirectRunner):** For local execution on the local machine:
        1. **Extract Embeddings with DirectRunner:** Execute the Apache Beam pipeline ([dataflow_embed.py](assets/dataflow_embed.py)) with `--runner=DirectRunner`, passing local audio paths for `--input_glob` and a local workspace directory (e.g., `agent_workspace/embeddings/<dataset>`) for `--output_dir`.
        1. **Convert to Hoplite DB:** Ingest the resulting Parquet embeddings into a Hoplite database in `databases/<dataset>` using [ingest_embeddings.py](assets/ingest_embeddings.py).
      - **Option B (Remote Ingestion via Colab):** For intermediate batches
        benefiting from GPU/TPU acceleration:
        1. **Stage Audio:** Transfer recordings to the Colab environment
           following the storage evaluation and upload protocols in
           `AGENTS.md`.
        1. **Remote Embedding with DirectRunner:** Run [dataflow_embed.py](assets/dataflow_embed.py)
           with `--runner=DirectRunner` on the Colab GPU/TPU runtime to generate
           sharded Apache Parquet embeddings.
        1. **Convert to Hoplite DB:** Ingest the Parquet embeddings into an ephemeral
           Hoplite database using [ingest_embeddings.py](assets/ingest_embeddings.py)
           and sync the database files into the local `databases/` directory. Ensure the database
           audio source metadata points to the local recordings path so the
           local web app can resolve and stream audio.
      - **Option C (Cloud-Scale Ingestion via GCP Dataflow):** For large
        recording corpora (hundreds of GBs to TBs):
        1. **Stage Audio to GCS:** Stage recordings to Cloud Storage
           following the GCP storage architecture and staging protocols in
           `AGENTS.md`.
        1. **Distributed Dataflow Embedding:** Execute the Apache Beam
           pipeline ([dataflow_embed.py](assets/dataflow_embed.py)) on
           Google Cloud Dataflow with `DataflowRunner` to extract embeddings and generate sharded
           Apache Parquet files on GCS.
        1. **Convert to Hoplite DB:** Ingest the resulting Parquet embeddings
           into a Hoplite database using [ingest_embeddings.py](assets/ingest_embeddings.py).
        1. **Mount with GCS FUSE:** Mount the audio bucket via GCS FUSE
           (`google-cloud-storage-fuse`) so the web app can stream audio
           windows on demand without downloading full recordings.
           Refer to [GCP Dataflow Audio Embedding](references/GCP_DATAFLOW_EMBEDDING.md)
           for complete instructions and commands.
1. **Build a Bioacoustics Web App:** Create an interactive webpage for the user
   to browse, search, and annotate audio snippets associated with the Hoplite
   database created in the previous step. Make sure the web app supports the
   following:
   - **Browsing:** Design the UI so that the user can inspect rows in the
     database and listen to their associated audio.
   - **Annotating:** Empower the user to attach annotations to rows in the
     database. Save, update, and clear user annotations (positive, negative,
     or uncertain) directly in the database under the "user" provenance tag
     as they interact.
   - **Searching:** Empower the user to reorder rows in the database
     according to various criteria:
     - **Vector Search:** Allow the user to present a search query in the
       form of a URI pointing to an audio clip. Embed it with the selected
       model and perform a search operation in the database. Use the result
       to rerank all rows in the database. Ensure that the query URI is
       only used for ranking, and any annotations submitted are saved under
       the active label (e.g., species name), NOT under the query URI
       itself.
     - **Trained Classifier**: Once enough annotations are provided for a
       particular label (at least two positives and one negative, or two
       negatives and one positive), allow the user to search with a
       classifier trained on those annotations. Train a linear classifier
       using `perch-hoplite` APIs, and use the classifier's weights to
       score database rows and rerank them.

## Technical Reference

For detailed API usage, implementation instructions, and code examples, see:

- [Bioacoustics Technical Reference](references/REFERENCE.md)
- [GCP Dataflow Audio Embedding Technical Reference](references/GCP_DATAFLOW_EMBEDDING.md)

This reference covers:

- Hoplite Database initialization and loading
- Audio embedding extraction via `dataflow_embed.py` (with `DirectRunner` locally or `DataflowRunner` on GCP)
- Ingesting Parquet embeddings into Hoplite DB using `ingest_embeddings.py`
- Distributed audio embedding on Google Cloud Dataflow with Apache Beam
- Ingesting Dataflow embeddings with `ingest_embeddings.py`
- GCS FUSE audio streaming without breaking changes
- Resolving physical audio files from database records
- Agile Modeling setup and search implementation
- Serving search results via the interactive UI
- Processing user annotations (saving, clearing, and restoring state)
