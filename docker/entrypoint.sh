#!/usr/bin/env sh
set -e

# Build args from env overrides, then append any CLI args.
EXTRA_ARGS=""

# Prefer explicit local dir (offline)
if [ -n "${SPAMBERT_LOCAL_MODEL_DIR}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --local-model-dir ${SPAMBERT_LOCAL_MODEL_DIR}"
fi

# Otherwise allow remote model override
if [ -n "${SPAMBERT_MODEL}" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --model ${SPAMBERT_MODEL}"
fi

# Use HF_HOME for cache to avoid TRANSFORMERS_CACHE deprecation
if [ -n "${HF_HOME}" ]; then
  export TRANSFORMERS_CACHE="${HF_HOME}/hub"
fi

# Launch spam_bert with any extras and passed-through args (CMD/CLI)
exec python -m spam_bert $EXTRA_ARGS "$@"
