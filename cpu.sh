#!/usr/bin/env bash
set -euo pipefail

CED_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare/conda_envs/ced_p0"
PY="$CED_ENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "[FATAL] Python not found: $PY" >&2
  exit 1
fi


add_lib_dir () {
  local d="$1"
  if [ -d "$d" ]; then
    if [[ ":${LD_LIBRARY_PATH:-}:" != *":$d:"* ]]; then
      export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  fi
}
add_lib_dir "$CED_ENV/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/torch/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/cusparse/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/cublas/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"


export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

MODEL_ID="${MODEL_ID:-auto}"
LOCAL_DIR="${LOCAL_DIR:-$PWD/../dataprepare/models/Qwen3-VL-8B-Instruct}"
CACHE_DIR="${CACHE_DIR:-$HF_HUB_CACHE}"
FALLBACK_REPO="${FALLBACK_REPO:-Qwen/Qwen3-VL-8B-Instruct}"

"$PY" -u main.py \
  --repo-id "$MODEL_ID" \
  --local-dir "$LOCAL_DIR" \
  --cache-dir "$CACHE_DIR" \
  --fallback-repo "$FALLBACK_REPO" \
  --download-only
