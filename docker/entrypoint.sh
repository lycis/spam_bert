#!/usr/bin/env sh
set -e

EXTRA_ARGS=""

# ------------------------------
# Model source selection
# ------------------------------
# Prefer explicit local dir (offline)
if [ -n "${SPAMBERT_LOCAL_MODEL_DIR}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --local-model-dir ${SPAMBERT_LOCAL_MODEL_DIR}"
fi

# Otherwise allow remote model override
if [ -n "${SPAMBERT_MODEL}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --model ${SPAMBERT_MODEL}"
fi

# Optional explicit cache dir for downloads (separate from HF_HOME)
if [ -n "${SPAMBERT_MODEL_CACHE_DIR}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --model-cache-dir ${SPAMBERT_MODEL_CACHE_DIR}"
fi

# ------------------------------
# Inference/threshold/aggregation
# ------------------------------
if [ -n "${SPAMBERT_THRESHOLD}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --threshold ${SPAMBERT_THRESHOLD}"
fi

if [ -n "${SPAMBERT_NO_CHUNK}" ]; then
  # Accept 1/true/TRUE/yes/YES
  case "$SPAMBERT_NO_CHUNK" in
    1|true|TRUE|yes|YES) EXTRA_ARGS="$EXTRA_ARGS --no-chunk" ;;
  esac
fi

if [ -n "${SPAMBERT_AGGREGATION}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --aggregate ${SPAMBERT_AGGREGATION}"
fi

if [ -n "${SPAMBERT_TOPK}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --topk ${SPAMBERT_TOPK}"
fi

if [ -n "${SPAMBERT_PER_CHUNK_THR}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --per-chunk-thr ${SPAMBERT_PER_CHUNK_THR}"
fi

if [ -n "${SPAMBERT_DECAY}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --decay ${SPAMBERT_DECAY}"
fi

# ------------------------------
# Server options
# ------------------------------
if [ -n "${SPAMBERT_HOST}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --host ${SPAMBERT_HOST}"
fi

if [ -n "${SPAMBERT_PORT}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --port ${SPAMBERT_PORT}"
fi

# ------------------------------
# Cache root (preferred modern var)
# ------------------------------
# Use HF_HOME for cache to avoid TRANSFORMERS_CACHE deprecation
# Transformers will use $HF_HOME/hub internally; we export TRANSFORMERS_CACHE
# too for libs that still read it.
if [ -n "${HF_HOME}" ]; then
  export TRANSFORMERS_CACHE="${HF_HOME}/hub"
fi

# Launch spam_bert with built env args, then append any CLI args from CMD.
# NOTE: CLI args override env-provided ones when they conflict.
exec python -m spam_bert $EXTRA_ARGS "$@"
