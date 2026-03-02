#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
LOCAL_DIR="${LOCAL_DIR:-./models/Qwen2.5-VL-7B-Instruct}"
CACHE_DIR="${CACHE_DIR:-./.hf_cache}"

# export CUDA_VISIBLE_DEVICES=0

python -u main.py \
  --repo-id "$MODEL_ID" \
  --local-dir "$LOCAL_DIR" \
  --cache-dir "$CACHE_DIR" \
  --device cuda:0 \
  --dtype auto \
  --smoke-test
