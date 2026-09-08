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

"""Universal Hoplite Database Ingestion Asset.

Converts sharded Apache Parquet embeddings into a queryable Perch Hoplite
database (SQLite metadata + USearch vector index), automatically rebinds audio
source paths, and performs sanity validation.
"""

import argparse
import json
import logging
import pathlib
from collections.abc import Iterator

import numpy as np
import soundfile
import tqdm
from etils import epath
from ml_collections import config_dict
from perch_hoplite.agile import embed, metadata, source_info
from perch_hoplite.db import db_loader, sqlite_usearch_impl


# Suppress internal SQL trace logs from perch-hoplite
class SQLSuppressFilter(logging.Filter):
    def filter(self, record):
        return "Executed SQL statement" not in record.getMessage()


logging.getLogger("absl").addFilter(SQLSuppressFilter())
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def iter_parquet_batches(
    embeddings_dir: epath.Path, batch_size: int = 1024
) -> Iterator[tuple[list[str], list[float], list[float], np.ndarray]]:
    """Yields (filenames, offsets_s, window_sizes_s, embeddings_matrix) from Parquet files."""
    import pyarrow.dataset as ds

    parquet_files = [f.as_posix() for f in embeddings_dir.glob("*.parquet")] or [
        f.as_posix() for f in embeddings_dir.glob("*/*.parquet")
    ]
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {embeddings_dir}")

    dataset = ds.dataset(parquet_files, format="parquet")
    schema = dataset.schema
    has_window_size = "window_size_s" in schema.names

    for batch in dataset.to_batches(batch_size=batch_size):
        filenames = batch.column("filename").to_pylist()
        offsets_s = batch.column("timestamp_s").to_pylist()
        if has_window_size:
            window_sizes_s = batch.column("window_size_s").to_pylist()
        else:
            window_sizes_s = [5.0] * len(filenames)

        # Convert fixed-size list or nested list of embeddings to contiguous 2D float32 numpy array
        raw_emb = batch.column("embedding").to_numpy(zero_copy_only=False)
        embeddings = np.array(
            [np.array(e, dtype=np.float32) for e in raw_emb], dtype=np.float32
        )
        yield filenames, offsets_s, window_sizes_s, embeddings


def ingest_embeddings(
    embeddings_path: str,
    db_path: str,
    audio_base_path: str,
    metadata_dir: str | None = None,
    dataset_name: str | None = None,
    batch_size: int = 1024,
    clean: bool = True,
    validate: bool = True,
) -> sqlite_usearch_impl.SQLiteUSearchDB:
    """Converts Parquet embeddings into a Hoplite DB and configures audio paths and metadata."""
    emb_dir = epath.Path(embeddings_path)
    dest_db_path = pathlib.Path(db_path)

    if dataset_name is None:
        dataset_name = dest_db_path.name

    parquet_files = [f.as_posix() for f in emb_dir.glob("*.parquet")] or [
        f.as_posix() for f in emb_dir.glob("*/*.parquet")
    ]
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {embeddings_path}.")

    logging.info(
        "Discovered %d Parquet files in %s", len(parquet_files), embeddings_path
    )

    import pyarrow.dataset as ds

    dataset = ds.dataset(parquet_files, format="parquet")

    # Read self-describing metadata directly from Parquet schema footer
    meta = dataset.schema.metadata or {}
    model_key = meta.get(b"model_key", b"taxonomy_model_tf").decode("utf-8")
    model_cfg_dict = {}
    if b"model_config" in meta:
        try:
            model_cfg_dict = json.loads(meta[b"model_config"].decode("utf-8"))
        except Exception:
            model_cfg_dict = {}

    # Clean existing DB target if requested
    if clean and dest_db_path.exists():
        logging.info("Cleaning existing database directory at %s...", dest_db_path)
        for p in dest_db_path.iterdir():
            if p.is_file():
                p.unlink()

    dest_db_path.mkdir(parents=True, exist_ok=True)

    batch_gen = iter_parquet_batches(emb_dir, batch_size=batch_size)
    try:
        first_batch = next(batch_gen)
    except StopIteration:
        raise ValueError(f"No embeddings found in {embeddings_path}.")

    emb_dim = first_batch[3].shape[-1]
    logging.info("Embedding dimension: %d", emb_dim)

    # Initialize Hoplite SQLite/USearch database
    db = db_loader.create_new_usearch_db(str(dest_db_path), emb_dim)

    # Load agile metadata if present
    agile_meta_dir = epath.Path(metadata_dir or audio_base_path)
    agile_md = metadata.AgileMetadata.from_directory(agile_meta_dir)

    # Register any extra schema fields from hoplite_metadata_description.csv
    builtin_cols = {
        "deployments": {"id", "name", "project", "latitude", "longitude", "deployment"},
        "recordings": {"id", "filename", "datetime", "deployment_id", "recording"},
    }
    dtype_map = {"str": str, "float": float, "int": int, "bytes": bytes}
    for field in agile_md.fields.values():
        table_name = (
            "deployments" if field.metadata_level == "deployment" else "recordings"
        )
        if field.field_name in builtin_cols.get(table_name, set()):
            continue
        col_type = dtype_map.get(field.dtype, str)
        try:
            db.add_extra_table_column(table_name, field.field_name, col_type)
        except Exception as exc:
            logging.debug(
                "Could not add extra column %s to %s: %s",
                field.field_name,
                table_name,
                exc,
            )

    # Configure and insert metadata
    audio_sources = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name=dataset_name,
                base_path=audio_base_path,
                file_glob="*/*.wav",
                min_audio_len_s=1.0,
                target_sample_rate_hz=-2,
                shard_len_s=None,
            ),
        )
    )
    db.insert_metadata("audio_sources", audio_sources.to_config_dict())

    model_config = embed.ModelConfig(
        model_key=model_key,
        embedding_dim=emb_dim,
        model_config=config_dict.ConfigDict(model_cfg_dict)
        if model_cfg_dict
        else config_dict.ConfigDict(),
    )
    db.insert_metadata("model_config", model_config.to_config_dict())

    deployment_id_map: dict[str, int] = {}
    recording_id_map: dict[str, int] = {}

    def get_or_insert_deployment(deployment_name: str) -> int:
        if deployment_name in deployment_id_map:
            return deployment_id_map[deployment_name]
        depl_kwargs = agile_md.get_deployment_metadata(deployment_name)
        depl_kwargs.pop("deployment", None)
        known_cols = db._extra_table_columns.get("deployments", {})
        filtered_kwargs = {
            k: v for k, v in depl_kwargs.items() if v is not None or k in known_cols
        }
        deployment_id = db.insert_deployment(
            name=deployment_name, project=dataset_name, **filtered_kwargs
        )
        deployment_id_map[deployment_name] = deployment_id
        return deployment_id

    total_windows = 0
    logging.info("Populating Hoplite database...")

    def process_batch(filenames, offsets, win_sizes, embeddings_matrix):
        nonlocal total_windows
        windows_batch = []
        for f_id, off, w_size in zip(filenames, offsets, win_sizes):
            if f_id in recording_id_map:
                rec_id = recording_id_map[f_id]
            else:
                # Infer deployment name from file hierarchy (matching EmbedWorker)
                depl_name = f_id.split("/")[0] if "/" in f_id else dataset_name
                depl_id = get_or_insert_deployment(depl_name)

                rec_kwargs = agile_md.get_recording_metadata(f_id)
                rec_kwargs.pop("recording", None)
                known_rec_cols = db._extra_table_columns.get("recordings", {})
                filtered_rec_kwargs = {
                    k: v
                    for k, v in rec_kwargs.items()
                    if v is not None or k in known_rec_cols
                }
                rec_id = db.insert_recording(
                    filename=f_id, deployment_id=depl_id, **filtered_rec_kwargs
                )
                recording_id_map[f_id] = rec_id
            windows_batch.append(
                {
                    "recording_id": rec_id,
                    "offsets": [off, off + w_size],
                }
            )
        db.insert_windows_batch(windows_batch, embeddings_matrix)
        total_windows += len(windows_batch)

    # Process first batch and all subsequent batches
    process_batch(*first_batch)
    for batch in tqdm.tqdm(batch_gen, desc="Ingesting batches"):
        process_batch(*batch)

    # Import annotations if present in agile metadata
    if agile_md.annotations:
        logging.info("Importing agile annotations...")
        num_annotations = 0
        for f_id, annotations in agile_md.annotations.items():
            if f_id not in recording_id_map:
                continue
            rec_id = recording_id_map[f_id]
            for ann in annotations:
                db.insert_annotation(
                    recording_id=rec_id,
                    offsets=ann.offsets,
                    label=ann.label,
                    label_type=ann.label_type,
                    provenance=ann.provenance or "agile_metadata",
                    handle_duplicates="allow",
                )
                num_annotations += 1
        logging.info("Successfully imported %d annotations.", num_annotations)

    db.commit()
    num_embeddings = db.count_embeddings()
    logging.info(
        "Successfully committed %d embeddings to %s", num_embeddings, dest_db_path
    )

    if validate:
        _validate_database(db, audio_base_path, dest_db_path)

    return db


def _validate_database(
    db: sqlite_usearch_impl.SQLiteUSearchDB,
    audio_base_path: str,
    db_path: pathlib.Path,
) -> None:
    """Runs validation sanity checks on database contents, audio slices, and search."""
    cursor = db._get_cursor()
    cursor.execute("SELECT COUNT(*) FROM deployments")
    num_deployments = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM recordings")
    num_recordings = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM windows")
    num_windows = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM annotations")
    num_annotations = cursor.fetchone()[0]
    total_embeddings = db.count_embeddings()

    logging.info("--- Database Summary ---")
    logging.info("Deployments : %d", num_deployments)
    logging.info("Recordings  : %d", num_recordings)
    logging.info("Windows     : %d", num_windows)
    logging.info("Annotations : %d", num_annotations)
    logging.info("Embeddings  : %d", total_embeddings)

    # Validate audio slice retrieval
    sample_window = db.get_window(1)
    sample_rec = db.get_recording(sample_window.recording_id)
    audio_path = pathlib.Path(audio_base_path) / sample_rec.filename
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio validation failed: file '{audio_path}' does not exist."
        )

    info = soundfile.info(str(audio_path))
    sr = info.samplerate
    start_sample = int(sample_window.offsets[0] * sr)
    end_sample = int(sample_window.offsets[1] * sr)
    data, _ = soundfile.read(str(audio_path), start=start_sample, stop=end_sample)
    logging.info(
        "Audio resolution verified: read %d samples (%s) at %d Hz",
        len(data),
        sample_rec.filename,
        sr,
    )

    # Validate vector search
    sample_emb = db.get_embedding(1)
    search_results = db.search(sample_emb, search_list_size=5)
    logging.info(
        "Vector search verified: query returned %d neighbors (top score: %.3f)",
        len(search_results.search_results),
        search_results.search_results[0].sort_score,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ingest sharded Parquet embeddings into a Hoplite vector database."
    )
    parser.add_argument(
        "--embeddings_path",
        "--embeddings_dir",
        dest="embeddings_path",
        type=str,
        required=True,
        help="Directory containing sharded .parquet embeddings.",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Target directory for the Hoplite database (e.g. databases/powdermill).",
    )
    parser.add_argument(
        "--audio_base_path",
        "--audio_dir",
        dest="audio_base_path",
        type=str,
        required=True,
        help="Base path to audio recordings directory (e.g. data/powdermill).",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default=None,
        help="Directory containing agile metadata CSV files (defaults to audio_base_path).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset name. Defaults to the name of the database directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for database insertions.",
    )
    parser.add_argument(
        "--no_clean",
        action="store_true",
        help="Do not clean existing database files in target directory.",
    )
    args = parser.parse_args()

    ingest_embeddings(
        embeddings_path=args.embeddings_path,
        db_path=args.db_path,
        audio_base_path=args.audio_base_path,
        metadata_dir=args.metadata_dir,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        clean=not args.no_clean,
        validate=True,
    )


if __name__ == "__main__":
    main()
