# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS app

# ---- Speed & reproducibility knobs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps kept minimal (no compilers if we can avoid source builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Copy only requirement files for better layer caching
COPY requirements.txt requirements-dev.txt* ./

# ---- Install PyTorch CPU wheels explicitly (super fast, no CUDA)
# Pin or pass via build args if you want stricter control
ARG TORCH_VERSION=2.3.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Use BuildKit pip cache; prefer binary wheels for everything
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
 && pip install --prefer-binary --index-url ${TORCH_INDEX_URL} torch==${TORCH_VERSION} \
 && pip install --prefer-binary -r requirements.txt uvicorn[standard]

# ---- Copy package (src layout) and metadata
COPY pyproject.toml setup.cfg README.md LICENSE* ./
COPY src ./src

# ---- Build a wheel, then install the wheel (faster than editable)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-deps -w dist . \
 && pip install --no-deps dist/*.whl

# ---- Entrypoint wrapper
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ---- Non-root
RUN useradd -ms /bin/bash appuser
USER appuser

EXPOSE 8000

# Prefer HF_HOME; models will cache under $HF_HOME/hub at runtime
ENV HF_HOME=/tmp/hf_home

ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
CMD ["--serve", "--host", "0.0.0.0", "--port", "8000"]
