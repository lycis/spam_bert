# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS app

# --- Speed & sanity
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# (Minimal) system deps; avoid compilers to prevent accidental source builds
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Install CPU-only PyTorch first
# -------------------------------
# Keep torch OUT of requirements-docker.txt.
# These args make bumping versions easy in CI.
ARG TORCH_VERSION=2.3.1
ARG TORCHVISION_VERSION=0.18.1
ARG TORCHAUDIO_VERSION=2.3.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Use BuildKit pip cache; prefer binary wheels everywhere
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
 && pip install --prefer-binary --index-url ${TORCH_INDEX_URL} \
      torch==${TORCH_VERSION} \
      torchvision==${TORCHVISION_VERSION} \
      torchaudio==${TORCHAUDIO_VERSION} \
      --extra-index-url https://pypi.org/simple

# -------------------------------
# App deps (docker-specific)
# -------------------------------
# requirements-docker.txt should NOT include torch/vision/audio
COPY requirements-docker.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary -r requirements-docker.txt

# -------------------------------
# App code (src layout) & metadata
# -------------------------------
COPY pyproject.toml setup.cfg README.md LICENSE* ./
COPY src ./src

# Build a wheel (faster and reproducible vs editable install) and install it
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-deps -w dist . \
 && pip install --no-deps dist/*.whl

# -------------------------------
# Entry point wrapper
# -------------------------------
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Non-root user
RUN useradd -ms /bin/bash appuser
USER appuser

# Model cache root (models will land under $HF_HOME/hub)
ENV HF_HOME=/tmp/hf_home

EXPOSE 8000

# Default: REST API on 0.0.0.0:8000
# Configure at runtime via env:
#   SPAMBERT_MODEL, SPAMBERT_LOCAL_MODEL_DIR, SPAMBERT_MODEL_CACHE_DIR,
#   SPAMBERT_THRESHOLD, SPAMBERT_NO_CHUNK, SPAMBERT_AGGREGATION,
#   SPAMBERT_TOPK, SPAMBERT_PER_CHUNK_THR, SPAMBERT_DECAY, SPAMBERT_HOST, SPAMBERT_PORT
ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
CMD ["--serve", "--host", "0.0.0.0", "--port", "8000"]
