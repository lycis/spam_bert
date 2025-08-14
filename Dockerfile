# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ----- Install Python deps first (better layer cache)
COPY requirements.txt ./
# Ensure uvicorn is present for --serve mode
RUN pip install --upgrade pip && pip install -r requirements.txt uvicorn[standard]

# ----- Copy package (src-layout) and metadata
COPY pyproject.toml setup.cfg README.md LICENSE* ./
COPY src ./src

# Install package into image (editable not needed in container, but fine)
RUN pip install --no-deps -e .

# ----- Entry point wrapper
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ----- Non-root user
RUN useradd -ms /bin/bash appuser
USER appuser

# Expose API port
EXPOSE 8000

# Prefer HF_HOME (Transformers v5+). Mount this to persist models.
ENV HF_HOME=/tmp/hf_home

# Default: serve API on 0.0.0.0:8000
# You can override model via:
#   - env: SPAMBERT_MODEL=prancyFox/tiny-bert-enron-spam
#   - env: SPAMBERT_LOCAL_MODEL_DIR=/models/tiny  (for offline)
# Or pass flags at runtime:  --model ...  --no-chunk  --threshold 0.55  etc.
ENTRYPOINT ["/bin/sh", "/app/entrypoint.sh"]
CMD ["--serve", "--host", "0.0.0.0", "--port", "8000"]
