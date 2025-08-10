# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first (better layer cache)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy package (src-layout) + entry
COPY pyproject.toml setup.cfg README.md LICENSE ./
COPY src ./src

# install package into image
RUN pip install --no-deps -e .

# non-root
RUN useradd -ms /bin/bash appuser
USER appuser

EXPOSE 8000
ENV TRANSFORMERS_CACHE=/app/.hf_cache

# default command: serve API
CMD ["python", "-m", "spam_bert", "--serve", "--host", "0.0.0.0", "--port", "8000"]
