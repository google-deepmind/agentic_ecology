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

"""Distributed Apache Beam pipeline for bioacoustics audio embedding.

This script executes an Apache Beam pipeline (locally with DirectRunner or at
scale on Google Cloud Dataflow) to extract windowed audio embeddings using Perch
models and serialize the results into sharded Apache Parquet files for downstream
ingestion into Perch Hoplite databases (`ingest_embeddings.py`).
"""

# Prevent dynamic library deadlock by importing tensorflow first on macOS
import tensorflow as tf  # noqa: F401 # isort: skip

import argparse
import json
import logging
from collections.abc import Iterable, Sequence
from typing import Any

import apache_beam as beam
import numpy as np
import pyarrow as pa
import soundfile
from apache_beam.io import parquetio
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
)
from etils import epath
from ml_collections import config_dict
from perch_hoplite import audio_io
from perch_hoplite.zoo import model_configs


def create_source_infos(
    source_file_patterns: Sequence[str],
    shard_len_s: float = -1.0,
) -> list[dict[str, Any]]:
    """Expands file glob patterns into a list of source info dictionaries."""
    source_files = []
    for pattern in source_file_patterns:
        if "**" in pattern:
            raise ValueError(
                f"Recursive wildcard '**' in pattern '{pattern}' is not supported by "
                "etils.epath on Cloud Storage. Use single-level wildcards such as '*/*.wav' instead."
            )

        if "*" in pattern:
            if "://" in pattern:
                scheme, rest = pattern.split("://", 1)
                p = epath.Path(f"{scheme}://")
            else:
                p = epath.Path(".")
                rest = pattern
            matched = list(p.glob(rest))
        else:
            p = epath.Path(pattern)
            matched = [p] if p.exists() else []
        source_files.extend(matched)

    logging.info("Found %d matching audio files.", len(source_files))
    source_infos = []
    for sf in source_files:
        path_str = sf.as_posix()
        if shard_len_s <= 0:
            source_infos.append(
                {"filepath": path_str, "shard_num": 0, "shard_len_s": -1.0}
            )
        else:
            try:
                with sf.open("rb") as f:
                    audio_file = soundfile.SoundFile(f)
                    duration_s = audio_file.frames / audio_file.samplerate
                num_shards = max(1, int(np.ceil(duration_s / shard_len_s)))
                for i in range(num_shards):
                    source_infos.append(
                        {
                            "filepath": path_str,
                            "shard_num": i,
                            "shard_len_s": shard_len_s,
                        }
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logging.warning(
                    "Could not read audio file header for %s: %s. Using single shard.",
                    path_str,
                    exc,
                )
                source_infos.append(
                    {"filepath": path_str, "shard_num": 0, "shard_len_s": -1.0}
                )

    return source_infos


class EmbedFn(beam.DoFn):
    """Beam DoFn to extract embeddings and yield structured records."""

    def __init__(
        self,
        model_key: str,
        model_config: dict[str, Any],
        file_id_depth: int = 1,
        min_audio_s: float = 1.0,
        target_sample_rate: int = -2,
        window_size_s: float = 5.0,
        hop_size_s: float = 5.0,
    ):
        """Initializes the EmbedFn DoFn."""
        self.model_key = model_key
        self.model_config = model_config
        self.file_id_depth = file_id_depth
        self.min_audio_s = min_audio_s
        self.target_sample_rate = target_sample_rate
        self.window_size_s = window_size_s
        self.hop_size_s = hop_size_s
        self.embedding_model = None

    def setup(self):
        """Initializes and caches the model on worker initialization."""
        logging.info("Setting up embedding model: %s", self.model_key)
        model_class = model_configs.get_model_class(self.model_key)
        cfg = config_dict.ConfigDict(self.model_config)
        self.embedding_model = model_class.from_config(cfg)

        if self.target_sample_rate == -2:
            self.target_sample_rate = self.embedding_model.sample_rate

    def process(self, source_info: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Processes a single source audio file or shard."""
        if self.embedding_model is None:
            self.setup()

        filepath = source_info["filepath"]
        shard_num = source_info.get("shard_num", 0)
        shard_len_s = source_info.get("shard_len_s", -1.0)

        parts = epath.Path(filepath).parts
        depth = min(self.file_id_depth + 1, len(parts))
        file_id = epath.Path(*parts[-depth:]).as_posix()

        try:
            audio_data = audio_io.load_audio(
                filepath,
                target_sample_rate=self.target_sample_rate,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.warning("Failed to load audio for %s: %s", filepath, exc)
            return

        if audio_data is None or len(audio_data) < int(
            self.min_audio_s * self.target_sample_rate
        ):
            logging.warning("Audio data too short or empty for %s.", filepath)
            return

        window_samples = int(self.window_size_s * self.target_sample_rate)
        hop_samples = int(self.hop_size_s * self.target_sample_rate)

        start_sample = 0
        if shard_len_s > 0:
            start_sample = int(shard_num * shard_len_s * self.target_sample_rate)
            end_sample = min(
                len(audio_data),
                int((shard_num + 1) * shard_len_s * self.target_sample_rate),
            )
            audio_data = audio_data[start_sample:end_sample]

        if len(audio_data) < window_samples:
            padded = np.zeros(window_samples, dtype=audio_data.dtype)
            padded[: len(audio_data)] = audio_data
            audio_data = padded

        num_windows = max(1, 1 + (len(audio_data) - window_samples) // hop_samples)
        for w_idx in range(num_windows):
            w_start = w_idx * hop_samples
            w_end = w_start + window_samples
            if w_end > len(audio_data):
                break

            chunk = audio_data[w_start:w_end]
            offset_s = float(start_sample + w_start) / float(self.target_sample_rate)

            try:
                outputs = self.embedding_model.embed(chunk)
                emb = np.squeeze(outputs.embeddings)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logging.warning(
                    "Inference failed for %s at %.2fs: %s", file_id, offset_s, exc
                )
                continue

            beam.metrics.Metrics.counter(
                "BioacousticsEmbedding", "windows_processed"
            ).inc()
            yield {
                "filename": file_id,
                "timestamp_s": float(offset_s),
                "window_size_s": float(self.window_size_s),
                "hop_size_s": float(self.hop_size_s),
                "embedding": emb.astype(np.float32).tolist(),
            }


def run_pipeline(argv: Sequence[str] | None = None) -> None:
    """Parses arguments and executes the Apache Beam embedding pipeline."""
    parser = argparse.ArgumentParser(description="Bioacoustic Audio Embedding Pipeline")
    parser.add_argument(
        "--input_glob",
        type=str,
        required=True,
        help="Glob pattern or comma-separated patterns for audio files (e.g., gs://bucket/audio/*/*.wav).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="GCS or local directory where embeddings will be saved.",
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default="perch_v2_cpu",
        help="Model preset key (e.g., perch_v2_cpu, perch_v2, surfperch).",
    )
    parser.add_argument(
        "--window_size_s",
        type=float,
        default=5.0,
        help="Window size in seconds.",
    )
    parser.add_argument(
        "--hop_size_s",
        type=float,
        default=5.0,
        help="Hop size in seconds.",
    )
    parser.add_argument(
        "--shard_len_s",
        type=float,
        default=-1.0,
        help="Duration in seconds to shard long audio files before embedding (-1.0 to disable).",
    )
    parser.add_argument(
        "--file_id_depth",
        type=int,
        default=1,
        help="Directory depth to preserve in output recording IDs.",
    )
    parser.add_argument(
        "--min_audio_s",
        type=float,
        default=1.0,
        help="Minimum audio length in seconds to process.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process a single sample locally and exit without launching Beam pipeline.",
    )

    known_args, beam_args = parser.parse_known_args(argv)

    source_patterns = [p.strip() for p in known_args.input_glob.split(",") if p.strip()]
    output_dir = epath.Path(known_args.output_dir)

    preset_cfg = model_configs.get_preset_model_config(known_args.model_key)
    model_config_dict = dict(preset_cfg.model_config)
    model_config_dict["window_size_s"] = known_args.window_size_s
    model_config_dict["hop_size_s"] = known_args.hop_size_s

    logging.info("Expanding source file patterns: %s", source_patterns)
    source_infos = create_source_infos(
        source_file_patterns=source_patterns,
        shard_len_s=known_args.shard_len_s,
    )
    if not source_infos:
        raise ValueError(f"No audio files found matching {known_args.input_glob}")

    if known_args.dry_run:
        logging.info("Starting dry-run test...")
        sample = source_infos[0]
        embed_fn = EmbedFn(
            model_key=preset_cfg.model_key,
            model_config=model_config_dict,
            file_id_depth=known_args.file_id_depth,
            min_audio_s=known_args.min_audio_s,
            window_size_s=known_args.window_size_s,
            hop_size_s=known_args.hop_size_s,
        )
        embed_fn.setup()
        results = list(embed_fn.process(sample))
        logging.info(
            "Dry run succeeded! Processed sample %s, generated %d windows.",
            sample["filepath"],
            len(results),
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PipelineOptions(beam_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    embed_fn = EmbedFn(
        model_key=preset_cfg.model_key,
        model_config=model_config_dict,
        file_id_depth=known_args.file_id_depth,
        min_audio_s=known_args.min_audio_s,
        window_size_s=known_args.window_size_s,
        hop_size_s=known_args.hop_size_s,
    )

    output_prefix = (output_dir / "embeddings").as_posix()
    logging.info("Starting Beam pipeline writing to %s...", output_prefix)

    metadata = {
        b"model_key": preset_cfg.model_key.encode("utf-8"),
        b"model_config": json.dumps(model_config_dict).encode("utf-8"),
        b"window_size_s": str(known_args.window_size_s).encode("utf-8"),
        b"hop_size_s": str(known_args.hop_size_s).encode("utf-8"),
        b"file_id_depth": str(known_args.file_id_depth).encode("utf-8"),
        b"source_file_patterns": json.dumps(source_patterns).encode("utf-8"),
    }

    parquet_schema = pa.schema(
        [
            ("filename", pa.string()),
            ("timestamp_s", pa.float64()),
            ("window_size_s", pa.float32()),
            ("hop_size_s", pa.float32()),
            ("embedding", pa.list_(pa.float32())),
        ],
        metadata=metadata,
    )

    with beam.Pipeline(options=pipeline_options) as pipeline:
        _ = (
            pipeline
            | "CreateSourceInfos" >> beam.Create(source_infos)
            | "ExtractEmbeddings" >> beam.ParDo(embed_fn)
            | "FilterEmpty" >> beam.Filter(lambda x: x is not None)
            | "Reshuffle" >> beam.Reshuffle()
            | "WriteParquet"
            >> parquetio.WriteToParquet(
                file_path_prefix=output_prefix,
                schema=parquet_schema,
                codec="zstd",
                file_name_suffix=".parquet",
            )
        )

    logging.info("Embedding pipeline completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
