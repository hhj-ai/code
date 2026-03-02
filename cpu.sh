#!/usr/bin/env bash
set -euo pipefail

# 直接用 conda 环境的绝对路径 python（不需要 conda activate）
CED_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare/conda_envs/ced_p0"
PY="$CED_ENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "[FATAL] Python not found or not executable: $PY" >&2
  echo "Check your conda env path: $CED_ENV" >&2
  exit 1
fi

# （可选）把 HF 缓存放到项目可控目录，避免写到 ~/.cache
export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
LOCAL_DIR="${LOCAL_DIR:-$PWD/models/Qwen2.5-VL-7B-Instruct}"
CACHE_DIR="${CACHE_DIR:-$HF_HUB_CACHE}"

"$PY" -u main.py \
  --repo-id "$MODEL_ID" \
  --local-dir "$LOCAL_DIR" \
  --cache-dir "$CACHE_DIR" \
  --device cpu \
  --dtype float32
