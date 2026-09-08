# GCP Dataflow Audio Embedding Technical Reference

This guide provides end-to-end technical instructions for scaling bioacoustic
audio embedding workloads to Google Cloud Platform (GCP) using Google Cloud
Storage (GCS) and Google Cloud Dataflow (Apache Beam).

## Overview & Architecture

When processing large collections of Passive Acoustic Monitoring (PAM) audio
(hundreds of gigabytes to multiple terabytes), local single-machine inference can
take days or weeks. This workflow offloads embedding extraction to Google Cloud
Dataflow, allowing horizontal distributed scaling across dozens or hundreds of
worker nodes.

```
┌────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Google Cloud Storage   │      │ Google Cloud Dataflow   │      │ Google Cloud Storage    │
│ gs://<BUCKET>/audio/   │ ───> │ Apache Beam Workers     │ ───> │ gs://<BUCKET>/          │
│ (.wav / .flac files)   │      │ (Perch embed models)    │      │   embeddings/<DATASET>/ │
└────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
                                                                              │
                                                                              ▼
┌────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Bioacoustics Web App   │      │ GCS FUSE Local Mount    │      │ Perch Hoplite Database  │
│ (Active Learning / UI) │ <─── │ Range-request streaming │ <─── │ ingest_embeddings.py    │
│ server.py              │      │ (zero file rewrites)    │      │ databases/<DATASET>/    │
└────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

The workflow consists of five stages:

1. **Prerequisites & GCP Setup**: Authenticate and configure Google Cloud tools.
1. **Bucket Provisioning & Audio Staging**: Organize and sync audio to Cloud Storage.
1. **Dataflow Embedding Pipeline**: Run the distributed Apache Beam pipeline.
1. **Hoplite Database Conversion**: Ingest output Parquet embeddings into a queryable Hoplite DB.
1. **Web App Audio Streaming with GCS FUSE**: Mount the GCS bucket to stream audio slices directly in the web UI.

______________________________________________________________________

## 1. Prerequisites & GCP Setup

### Install Required Google Skills

Ensure the relevant official skills from `google/skills` are installed in the
project workspace (as configured by `agentic-ecology-init`):

```bash
npx skills add google/skills --skill gcloud --skill google-cloud-storage-basics --skill google-cloud-storage-bucket-architect --skill google-cloud-storage-fuse --skill cloud-logging-query-generation -y -a <agent_name>
```

### Pipeline Dependencies

Apache Beam with GCP extras (`apache-beam[gcp]`) is bundled directly in the project's
reference dependencies (`pyproject.toml` / `uv.lock`) and is automatically installed
into the local environment when executing `uv sync` during project initialization.

### Authentication & Project Configuration

Authenticate using the Google Cloud CLI (following guidelines in the `gcloud` skill):

```bash
# Login with user credentials and configure Application Default Credentials (ADC)
gcloud auth login --quiet
gcloud auth application-default login --quiet

# Configure target project and compute region
gcloud config set project <PROJECT_ID> --quiet
gcloud config set compute/region <REGION> --quiet
```

### Enable Required Google Cloud APIs

Ensure the necessary Google Cloud APIs are enabled for your project:

```bash
gcloud services enable \
  dataflow.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  --project=<PROJECT_ID> --quiet
```

______________________________________________________________________

## 2. Bucket Architecture & Audio Staging

### Bucket Design & Creation

Use the `google-cloud-storage-bucket-architect` and `google-cloud-storage-basics`
skills to create a regional Cloud Storage bucket matching your compute region:

```bash
CLOUDSDK_METRICS_ENVIRONMENT="gcs-skills gcs-skills/1.0 (skill:google-cloud-storage-basics)" \
gcloud storage buckets create gs://<BUCKET_NAME> \
  --project=<PROJECT_ID> \
  --location=<REGION> \
  --uniform-bucket-level-access \
  --quiet
```

### Recommended Storage Layout

Structure your Cloud Storage bucket using the following directory layout:

```text
gs://<BUCKET_NAME>/
├── audio/<DATASET_NAME>/          # Raw recording files (.wav, .flac)
├── embeddings/<DATASET_NAME>/     # Sharded Parquet files (*.parquet)
├── staging/                       # Dataflow pipeline binary staging
└── temp/                          # Dataflow temporary files during job execution
```

### Staging Audio Recordings

Upload recordings to Cloud Storage using parallel, multi-threaded transfer:

```bash
# Sync local audio dataset directory to GCS
gcloud storage rsync -r ./data/<DATASET_NAME> gs://<BUCKET_NAME>/audio/<DATASET_NAME> \
  --project=<PROJECT_ID> --quiet
```

______________________________________________________________________

## 3. Dataflow Embedding Pipeline

The reusable pipeline script is located at:
`skills/agentic-ecology-bioacoustics/assets/dataflow_embed.py`

### Pipeline Capabilities

- **Multi-Runner Support**: Executes with `--runner=DirectRunner` for local CPU testing
  or GPU acceleration in Colab, or `--runner=DataflowRunner` for distributed horizontal
  scaling across Google Cloud Dataflow worker pools.
- **Model Preset Selection**: Uses `perch_hoplite.zoo.model_configs` to instantiate
  embedding models (e.g., `perch_v2_cpu`, `perch_v2`, `surfperch`).
- **Distributed Inference**: Workers load model weights once during `setup()`,
  slice audio files into uniform windows (e.g. 5.0s with 5.0s hop), compute embeddings,
  and write sharded Apache Parquet files (`embeddings-*.parquet`).
- **Self-Describing Metadata**: Automatically embeds model parameters, hop/window
  sizes, and file patterns directly inside the Parquet schema metadata footer, eliminating
  the need for separate sidecar configuration files.

### Dry-Run Validation (Local)

Before submitting a large distributed job to Dataflow, validate your configuration
and model instantiation locally using `DirectRunner` with the `--dry_run` flag:

```bash
uv run python skills/agentic-ecology-bioacoustics/assets/dataflow_embed.py \
  --input_glob="gs://<BUCKET_NAME>/audio/<DATASET_NAME>/*/*.wav" \
  --output_dir="scratch/dry_run_embeddings" \
  --model_key="perch_v2_cpu" \
  --dry_run
```

### Building the Dataflow Worker Container

Dataflow workers require system audio decoding libraries (`libsndfile1`), ML packages
(`tensorflow`, `perch-hoplite`, `pyarrow`), and pre-cached model weights. Build and push
the worker image to Google Artifact Registry using Google Cloud Build with Kaniko:

```bash
# Build and push the worker image via Cloud Build with Kaniko
gcloud builds submit \
  --config=skills/agentic-ecology-bioacoustics/assets/cloudbuild.yaml \
  --substitutions=_IMAGE_TAG="<REGION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/perch-worker:latest" \
  --project="<PROJECT_ID>" \
  --quiet \
  skills/agentic-ecology-bioacoustics/assets/
```

### Submitting to Google Cloud Dataflow

Submit the distributed job to Dataflow with `DataflowRunner` pointing to the pre-built worker image.
Note: Use single-level wildcards (such as `*/*.wav`) because `etils.epath` on Cloud Storage does not
support recursive `**` patterns:

```bash
uv run python skills/agentic-ecology-bioacoustics/assets/dataflow_embed.py \
  --input_glob="gs://<BUCKET_NAME>/audio/<DATASET_NAME>/*/*.wav" \
  --output_dir="gs://<BUCKET_NAME>/embeddings/<DATASET_NAME>" \
  --output_format="parquet" \
  --model_key="perch_v2_cpu" \
  --window_size_s=5.0 \
  --hop_size_s=5.0 \
  --runner="DataflowRunner" \
  --project="<PROJECT_ID>" \
  --region="<REGION>" \
  --temp_location="gs://<BUCKET_NAME>/temp" \
  --staging_location="gs://<BUCKET_NAME>/staging" \
  --machine_type="n1-standard-4" \
  --max_num_workers=32 \
  --sdk_container_image="<REGION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/perch-worker:latest"
```

______________________________________________________________________

## 4. Monitoring & Troubleshooting

### Tracking Job Status

Inspect running Dataflow jobs using the `gcloud` skill:

```bash
# List active Dataflow jobs in the project
gcloud dataflow jobs list --project=<PROJECT_ID> --region=<REGION> --filter="state:JOB_STATE_RUNNING" --quiet

# Inspect specific job execution details
gcloud dataflow jobs describe <JOB_ID> --project=<PROJECT_ID> --region=<REGION> --quiet
```

### Diagnosing Worker Errors with Cloud Logging

Use the `cloud-logging-query-generation` skill to query worker stderr and exception
logs if a pipeline stage fails:

```bash
gcloud logging read \
  'resource.type="dataflow_step" AND resource.labels.job_id="<JOB_ID>" AND severity>=ERROR' \
  --project=<PROJECT_ID> \
  --limit=20 \
  --format=json \
  --quiet
```

______________________________________________________________________

## 5. Ingesting Cloud Embeddings into Hoplite

Once the Dataflow job finishes, the output directory (`gs://<BUCKET_NAME>/embeddings/<DATASET_NAME>`)
contains self-describing sharded `embeddings-*.parquet` files.

Use the turnkey ingestion script ([`ingest_embeddings.py`](../assets/ingest_embeddings.py)) to
construct the queryable Hoplite database (SQLite metadata + USearch vector index), automatically
rebind the audio path, and run validation sanity checks:

```bash
uv run python skills/agentic-ecology-bioacoustics/assets/ingest_embeddings.py \
  --embeddings_path="gs://<BUCKET_NAME>/embeddings/<DATASET_NAME>" \
  --db_path="databases/<DATASET_NAME>" \
  --audio_base_path="data/<DATASET_NAME>"
```

> [!NOTE]
> If streaming audio via GCS FUSE without downloading files locally, set `--audio_base_path="data/gcs_mount/audio/<DATASET_NAME>"`.

The ingestion script:

1. Lazily reads sharded Parquet batches using `pyarrow.dataset` (with zero TensorFlow dependencies).
1. Initializes the Hoplite database with the matching embedding dimension (e.g. 1536).
1. Rebinds `audio_sources` metadata to `--audio_base_path` so the Web App never encounters `FileNotFoundError`.
1. Validates audio slice reading via `soundfile` and runs a test vector search query against the USearch index.

This populates:

- The USearch index with window embedding vectors.
- SQLite tables with `Window`, `Recording`, `Deployment`, and `model_config` metadata.

______________________________________________________________________

## 6. Audio Streaming with GCS FUSE

To interact with the database using the Bioacoustics Web App without downloading
the entire audio corpus locally, mount the GCS bucket using **GCS FUSE**
(`google/skills@google-cloud-storage-fuse`).

### Why GCS FUSE Preserves the Streaming Paradigm

In the Bioacoustics Web App (`server.py`), audio playback requests (`/stream`)
invoke `_read_audio_window(db, window_id)`. `soundfile` opens the file path and seeks
directly to the window offsets.

When the bucket is mounted via GCS FUSE:

1. The GCS bucket appears as a local filesystem directory.
1. When `soundfile` seeks and reads the 5-second slice, GCS FUSE issues **HTTP range requests**
   under the hood to fetch only that specific chunk of audio from Google Cloud Storage.
1. The server converts this 5-second chunk to WAV bytes in-memory and streams it to
   the user's browser.
1. **No large audio files are ever downloaded in full**, completely preserving the
   existing low-latency streaming paradigm with zero modifications to `server.py`.

### Mounting the GCS Bucket

```bash
# Create local mount point directory
mkdir -p data/gcs_mount

# Mount the audio bucket using gcsfuse with implicit directories and fast read caching
gcsfuse --implicit-dirs <BUCKET_NAME> data/gcs_mount
```

When creating or converting the Hoplite database, set the dataset `base_path` in
`AudioSourceConfig` to `data/gcs_mount/audio/<DATASET_NAME>`. The Web App will
instantly resolve and stream all audio snippets on demand.

### Unmounting when Finished

```bash
# macOS
umount data/gcs_mount

# Linux
fusermount -u data/gcs_mount
```
