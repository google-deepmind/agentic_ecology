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

"""Environment verification script for agentic ecology workspaces."""

import logging
import sys

# Rule 1: Always import TensorFlow first on macOS to avoid dynamic library deadlock
# with PyArrow / Abseil symbols.
import tensorflow as tf

# Rule 2: Suppress internal perch-hoplite SQL trace logs
class SQLSuppressFilter(logging.Filter):
  """Filter to suppress excessive SQL trace logs from perch-hoplite."""

  def filter(self, record: logging.LogRecord) -> bool:
    return "Executed SQL statement" not in record.getMessage()

logging.getLogger("absl").addFilter(SQLSuppressFilter())

# Verify core dependencies
import perch_hoplite
import soundfile
import speciesnet


def main() -> None:
  print("✓ All agentic ecology packages loaded successfully.")
  print(f"  Python executable: {sys.executable}")
  print(f"  Perch Hoplite: {getattr(perch_hoplite, '__version__', 'available')}")
  print(f"  SoundFile version: {soundfile.__version__}")
  print(f"  SpeciesNet: {getattr(speciesnet, '__version__', 'available')}")
  print(f"  TensorFlow version: {tf.__version__}")


if __name__ == "__main__":
  main()
