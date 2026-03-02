    #!/usr/bin/env bash
    set -euo pipefail

    # 强制使用 conda 环境的绝对路径 python（不依赖 conda activate）
    CED_ENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare/conda_envs/ced_p0"
    PY="$CED_ENV/bin/python"

    if [ -z "${BASH_VERSION:-}" ]; then
      echo "[FATAL] 请用 bash 运行：bash gpu.sh（不要用 sh gpu.sh）" >&2
      exit 2
    fi

    if [ ! -x "$PY" ]; then
      echo "[FATAL] Python not found or not executable: $PY" >&2
      exit 1
    fi


# Build LD_LIBRARY_PATH so that conda env bundled NVIDIA libs are preferred.
# This often fixes: libcusparse.so.12 ... __nvJitLinkComplete_12_4 ... not defined ...
add_lib_dir () {
  local d="$1"
  if [ -d "$d" ]; then
    if [[ ":${LD_LIBRARY_PATH:-}:" != *":$d:"* ]]; then
      export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  fi
}

# 1) basic conda env libs
add_lib_dir "$CED_ENV/lib"

# 2) torch bundled libs
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/torch/lib"

# 3) pip nvidia libs (names used by PyTorch wheels)
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/cusparse/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/cublas/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"


    # （可选）把 HF 缓存放到项目可控目录
    export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
    export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

    MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
    LOCAL_DIR="${LOCAL_DIR:-$PWD/models/Qwen2.5-VL-7B-Instruct}"
    CACHE_DIR="${CACHE_DIR:-$HF_HUB_CACHE}"

    # 可选：指定 GPU
    # export CUDA_VISIBLE_DEVICES=0

    "$PY" -u main.py \
      --repo-id "$MODEL_ID" \
      --local-dir "$LOCAL_DIR" \
      --cache-dir "$CACHE_DIR" \
      --device cuda:0 \
      --dtype auto \
      --smoke-test
