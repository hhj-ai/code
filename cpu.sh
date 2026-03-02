    #!/usr/bin/env bash
    set -euo pipefail

    # ✅ 不用 conda activate，直接用 env 的 python 绝对路径
    CED_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare/conda_envs/ced_p0"
    PY="$CED_ENV/bin/python"
    if [ ! -x "$PY" ]; then
      echo "[FATAL] Python not found: $PY" >&2
      exit 1
    fi

    # ✅ 让 env 内 CUDA 相关 so 优先生效（避免系统库抢先被加载）

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


    # ✅ HF 缓存建议放到项目目录（可控、可迁移）
    export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

    # ✅ 默认自动找本地 Qwen3-VL（找不到才下载 fallback）
    MODEL_ID="${MODEL_ID:-auto}"
    LOCAL_DIR="${LOCAL_DIR:-$PWD/../dataprepare/models/Qwen3-VL-8B-Instruct}"
    CACHE_DIR="${CACHE_DIR:-$HF_HUB_CACHE}"
    FALLBACK_REPO="${FALLBACK_REPO:-Qwen/Qwen3-VL-8B-Instruct}"

    # 只下载不加载（规避 torch/CUDA so 问题）：DOWNLOAD_ONLY=1 bash cpu.sh
    if [ "${DOWNLOAD_ONLY:-0}" = "1" ]; then
      "$PY" -u main.py \
        --repo-id "$MODEL_ID" \
        --local-dir "$LOCAL_DIR" \
        --cache-dir "$CACHE_DIR" \
        --fallback-repo "$FALLBACK_REPO" \
        --download-only
      exit 0
    fi

    "$PY" -u main.py \
      --repo-id "$MODEL_ID" \
      --local-dir "$LOCAL_DIR" \
      --cache-dir "$CACHE_DIR" \
      --fallback-repo "$FALLBACK_REPO" \
      --device cpu \
      --dtype float32
