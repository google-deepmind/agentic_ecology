# Apache Beam Python 3.12 Dataflow worker image with bioacoustic dependencies
FROM apache/beam_python3.12_sdk:2.76.0

# Install system dependencies (libsndfile for audio I/O, git for source installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install bioacoustics ML dependencies
RUN pip install --no-cache-dir \
    tensorflow==2.21.0 \
    soundfile==0.14.0 \
    ml-collections==0.1.1 \
    etils[epath]==1.14.0 \
    kagglehub==0.3.13 \
    "git+https://github.com/google-research/perch-hoplite.git"

# Pre-download and cache model weights into image
RUN python3 -c "import tensorflow as tf; from perch_hoplite.zoo import model_configs; preset = model_configs.get_preset_model_config('perch_v2_cpu'); cls = model_configs.get_model_class(preset.model_key); cls.from_config(preset.model_config)" \
    && chmod -R a+rX /root/.cache

ENTRYPOINT ["/opt/apache/beam/boot"]
