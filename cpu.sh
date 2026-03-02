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

    # ✅ v8 改动：CPU 默认只做下载/复用检查，避免 8B fp32 直接 OOM 被 kill
    if [ "${RUN_CPU_LOAD:-0}" != "1" ]; then
      echo "[INFO] CPU 默认不加载 8B 模型（容易 OOM 被 kill）。如需强制 CPU 加载，设置 RUN_CPU_LOAD=1。" >&2
      "$PY" -u main.py \
        --repo-id "$MODEL_ID" \
        --local-dir "$LOCAL_DIR" \
        --cache-dir "$CACHE_DIR" \
        --fallback-repo "$FALLBACK_REPO" \
        --download-only
      exit 0
    fi

    # ✅ 强制 CPU 加载：默认 bfloat16 + 可选磁盘 offload
    CPU_DTYPE="${CPU_DTYPE:-bfloat16}"
    # 例如：MAX_CPU_MEM=24GiB OFFLOAD_FOLDER=$PWD/.offload_cpu RUN_CPU_LOAD=1 bash cpu.sh
    MAX_CPU_MEM="${MAX_CPU_MEM:-}"
    OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-$PWD/.offload_cpu}"

    EXTRA=()
    if [ -n "$MAX_CPU_MEM" ]; then
      EXTRA+=(--device-map auto --max-cpu-mem "$MAX_CPU_MEM" --offload-folder "$OFFLOAD_FOLDER")
    fi

    "$PY" -u main.py \
      --repo-id "$MODEL_ID" \
      --local-dir "$LOCAL_DIR" \
      --cache-dir "$CACHE_DIR" \
      --fallback-repo "$FALLBACK_REPO" \
      --device cpu \
      --dtype "$CPU_DTYPE" \
      "${EXTRA[@]}"
