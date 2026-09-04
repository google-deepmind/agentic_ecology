# coding=utf-8
# Copyright 2026 The Agentic Ecology Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Distributed Apache Beam pipeline for bioacoustics audio embedding.

This script executes an Apache Beam pipeline (locally with DirectRunner or at
scale on Google Cloud Dataflow) to extract windowed audio embeddings using Perch
models and serialize the results into sharded TFRecords compatible with Perch
Hoplite database conversion (`convert_legacy.convert_tfrecords`).
"""

import argparse
import dataclasses
import json
import logging
import os
import sys
from typing import Any, Iterable, Sequence

import apache_beam as beam
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import WorkerOptions
from etils import epath
from ml_collections import config_dict
import numpy as np
from perch_hoplite import audio_io
from perch_hoplite.zoo import model_configs
from perch_hoplite.zoo import zoo_interface
import soundfile
import tensorflow as tf


@dataclasses.dataclass
class SourceInfo:
  """Source information for an audio file."""

  filepath: str
  shard_num: int = 0
  shard_len_s: float = -1.0

  def file_id(self, file_id_depth: int) -> str:
    """Extracts a relative file ID based on directory depth."""
    parts = epath.Path(self.filepath).parts
    depth = min(file_id_depth + 1, len(parts))
    return epath.Path(*parts[-depth:]).as_posix()


def create_source_infos(
    source_file_patterns: Sequence[str],
    shard_len_s: float = -1.0,
) -> list[SourceInfo]:
  """Expands file glob patterns into a list of SourceInfo objects."""
  source_files = []
  for pattern in source_file_patterns:
    if "://" in pattern:
      scheme, rest = pattern.split("://", 1)
      root = f"{scheme}://"
    else:
      root = ""
      rest = pattern

    p = epath.Path(root)
    matched = list(p.glob(rest))
    source_files.extend(matched)

  logging.info("Found %d matching audio files.", len(source_files))
  source_infos = []
  for sf in source_files:
    path_str = sf.as_posix()
    if shard_len_s <= 0:
      source_infos.append(SourceInfo(filepath=path_str, shard_num=0, shard_len_s=-1.0))
    else:
      try:
        with sf.open("rb") as f:
          audio_file = soundfile.SoundFile(f)
          duration_s = audio_file.frames / audio_file.samplerate
        num_shards = max(1, int(np.ceil(duration_s / shard_len_s)))
        for i in range(num_shards):
          source_infos.append(
              SourceInfo(filepath=path_str, shard_num=i, shard_len_s=shard_len_s)
          )
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.warning("Could not read audio file header for %s: %s. Using single shard.", path_str, exc)
        source_infos.append(SourceInfo(filepath=path_str, shard_num=0, shard_len_s=-1.0))

  return source_infos


class EmbedFn(beam.DoFn):
  """Beam DoFn to extract embeddings and serialize to TFExample protos."""

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
    self.embedding_model: zoo_interface.EmbeddingModel | None = None

  def setup(self):
    """Instantiates the model once per worker."""
    logging.info("Setting up embedding model: %s", self.model_key)
    cfg = config_dict.ConfigDict(self.model_config)
    model_class = model_configs.get_model_class(self.model_key)
    self.embedding_model = model_class.from_config(cfg)
    if self.target_sample_rate == -2:
      self.target_sample_rate = self.embedding_model.sample_rate

  def process(self, source_info: SourceInfo) -> Iterable[tf.train.Example]:
    """Processes a single source audio file or shard."""
    if self.embedding_model is None:
      self.setup()

    file_id = source_info.file_id(self.file_id_depth)
    try:
      audio_data = audio_io.load_audio(
          source_info.filepath,
          target_sample_rate=self.target_sample_rate,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.warning("Failed to load audio for %s: %s", source_info.filepath, exc)
      return

    if audio_data is None or len(audio_data) < int(self.min_audio_s * self.target_sample_rate):
      return

    if audio_data.ndim > 1:
      audio_data = np.mean(audio_data, axis=-1)

    window_samples = int(self.window_size_s * self.target_sample_rate)
    hop_samples = int(self.hop_size_s * self.target_sample_rate)

    start_sample = 0
    if source_info.shard_len_s > 0:
      start_sample = int(source_info.shard_num * source_info.shard_len_s * self.target_sample_rate)
      end_sample = min(
          len(audio_data),
          int((source_info.shard_num + 1) * source_info.shard_len_s * self.target_sample_rate),
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
        logging.warning("Inference failed for %s at %.2fs: %s", file_id, offset_s, exc)
        continue

      feature = {
          "filename": tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[file_id.encode("utf-8")])
          ),
          "timestamp_s": tf.train.Feature(
              float_list=tf.train.FloatList(value=[offset_s])
          ),
          "embedding": tf.train.Feature(
              bytes_list=tf.train.BytesList(
                  value=[tf.io.serialize_tensor(emb.astype(np.float32)).numpy()]
              )
          ),
          "embedding_shape": tf.train.Feature(
              int64_list=tf.train.Int64List(value=list(emb.shape))
          ),
      }

      example_proto = tf.train.Example(
          features=tf.train.Features(feature=feature)
      )
      beam.metrics.Metrics.counter("BioacousticsEmbedding", "windows_processed").inc()
      yield example_proto


def write_metadata_config(
    output_dir: epath.Path,
    source_file_patterns: Sequence[str],
    model_key: str,
    model_config: dict[str, Any],
    file_id_depth: int,
    min_audio_s: float,
    target_sample_rate: int,
) -> None:
  """Writes config.json expected by convert_legacy.convert_tfrecords."""
  config_data = {
      "source_file_patterns": list(source_file_patterns),
      "output_dir": output_dir.as_posix(),
      "embed_fn_config": {
          "model_key": model_key,
          "model_config": model_config,
          "file_id_depth": file_id_depth,
          "min_audio_s": min_audio_s,
          "target_sample_rate_hz": target_sample_rate,
      },
  }
  config_path = output_dir / "config.json"
  with config_path.open("w") as f:
    f.write(json.dumps(config_data, indent=2))
  logging.info("Wrote pipeline config to %s", config_path.as_posix())


def run_pipeline(argv: Sequence[str] | None = None) -> None:
  """Parses arguments and executes the Apache Beam embedding pipeline."""
  parser = argparse.ArgumentParser(description="Bioacoustic Audio Embedding Pipeline")
  parser.add_argument(
      "--input_glob",
      type=str,
      required=True,
      help="Glob pattern or comma-separated patterns for audio files (e.g., gs://bucket/audio/**/*.wav).",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Destination directory for embeddings and config.json (e.g., gs://bucket/embeddings/dataset).",
  )
  parser.add_argument(
      "--model_key",
      type=str,
      default="perch_v2",
      help="Model preset identifier from perch_hoplite.zoo.model_configs (e.g., perch_v2, surfperch).",
  )
  parser.add_argument(
      "--window_size_s",
      type=float,
      default=5.0,
      help="Window length in seconds for audio segments.",
  )
  parser.add_argument(
      "--hop_size_s",
      type=float,
      default=5.0,
      help="Hop size in seconds between consecutive windows.",
  )
  parser.add_argument(
      "--file_id_depth",
      type=int,
      default=1,
      help="Number of parent directory components to keep in the recording filename.",
  )
  parser.add_argument(
      "--min_audio_s",
      type=float,
      default=1.0,
      help="Minimum audio file duration in seconds.",
  )
  parser.add_argument(
      "--shard_len_s",
      type=float,
      default=-1.0,
      help="Length in seconds for sharding long files (or -1 to process whole files).",
  )
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Test model instantiation and process a single sample without launching full pipeline.",
  )

  known_args, beam_args = parser.parse_known_args(argv)

  source_patterns = [p.strip() for p in known_args.input_glob.split(",") if p.strip()]
  output_dir = epath.Path(known_args.output_dir)

  preset_cfg = model_configs.get_model_config(known_args.model_key)
  model_config_dict = dict(preset_cfg.model_config)
  model_config_dict["window_size_s"] = known_args.window_size_s
  model_config_dict["hop_size_s"] = known_args.hop_size_s

  logging.info("Expanding source file patterns: %s", source_patterns)
  source_infos = create_source_infos(
      source_patterns, shard_len_s=known_args.shard_len_s
  )
  if not source_infos:
    logging.error("No matching audio files found for patterns: %s", source_patterns)
    sys.exit(1)

  if known_args.dry_run:
    logging.info("Starting dry-run test...")
    sample = source_infos[0]
    embed_fn = EmbedFn(
        model_key=known_args.model_key,
        model_config=model_config_dict,
        file_id_depth=known_args.file_id_depth,
        min_audio_s=known_args.min_audio_s,
        window_size_s=known_args.window_size_s,
        hop_size_s=known_args.hop_size_s,
    )
    embed_fn.setup()
    results = list(embed_fn.process(sample))
    logging.info(
        "Dry run succeeded! Processed sample %s, generated %d TFExamples.",
        sample.filepath,
        len(results),
    )
    return

  output_dir.mkdir(parents=True, exist_ok=True)
  write_metadata_config(
      output_dir=output_dir,
      source_file_patterns=source_patterns,
      model_key=known_args.model_key,
      model_config=model_config_dict,
      file_id_depth=known_args.file_id_depth,
      min_audio_s=known_args.min_audio_s,
      target_sample_rate=-2,
  )

  pipeline_options = PipelineOptions(beam_args)
  pipeline_options.view_as(SetupOptions).save_main_session = True

  embed_fn = EmbedFn(
      model_key=known_args.model_key,
      model_config=model_config_dict,
      file_id_depth=known_args.file_id_depth,
      min_audio_s=known_args.min_audio_s,
      window_size_s=known_args.window_size_s,
      hop_size_s=known_args.hop_size_s,
  )

  output_prefix = (output_dir / "embeddings").as_posix()
  logging.info("Starting Beam pipeline writing to %s...", output_prefix)

  with beam.Pipeline(options=pipeline_options) as pipeline:
    _ = (
        pipeline
        | "CreateSourceInfos" >> beam.Create(source_infos)
        | "ExtractEmbeddings" >> beam.ParDo(embed_fn)
        | "FilterEmpty" >> beam.Filter(lambda x: x is not None)
        | "Reshuffle" >> beam.Reshuffle()
        | "WriteTFRecords"
        >> beam.io.tfrecordio.WriteToTFRecord(
            file_path_prefix=output_prefix,
            coder=beam.coders.ProtoCoder(tf.train.Example),
            file_name_suffix=".tfrec",
        )
    )

  logging.info("Embedding pipeline completed successfully.")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  run_pipeline()
