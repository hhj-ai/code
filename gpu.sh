#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# gpu.sh (v11)
# 默认：跑“整个数据集”并产出结果（predictions.jsonl + summary.json）
#
# ✅ GPU 机无网：强制离线 + 只用本地模型/数据
# ✅ 仍保留：
#    - --smoke：只验活（加载+短生成）
#    - 自定义命令：bash gpu.sh python xxx.py --model_path {MODEL_PATH} ...
#
# 默认数据集定位：
#   1) 你显式传入：DATA_PATH=/path/to/data.jsonl bash gpu.sh
#   2) 或者：bash gpu.sh --data /path/to/data.jsonl
#   3) 都不传：自动在 ../dataprepare/data / ../dataprepare/datasets 等位置找最新的 *.jsonl/*.json
#
# 默认输出：
#   ./logs/dataset_run_<ts>/predictions.jsonl
#   ./logs/dataset_run_<ts>/summary.json
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CED_ENV="${CED_ENV:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare/conda_envs/ced_p0}"
PY="${PYTHON_BIN:-$CED_ENV/bin/python}"

if [ ! -x "$PY" ]; then
  echo "[FATAL] Python not found: $PY" >&2
  echo "        先在 CPU 机把 conda env 建好/装好包，然后 GPU 机再跑本脚本。" >&2
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

# ---- HF 离线缓存 + 强制离线 ----
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

MODEL_ID="${MODEL_ID:-auto}"
LOCAL_DIR="${LOCAL_DIR:-$SCRIPT_DIR/../dataprepare/models/Qwen3-VL-8B-Instruct}"
CACHE_DIR="${CACHE_DIR:-$HF_HUB_CACHE}"
FALLBACK_REPO="${FALLBACK_REPO:-Qwen/Qwen3-VL-8B-Instruct}"

DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-auto}"

mkdir -p "$SCRIPT_DIR/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$SCRIPT_DIR/logs/gpu_${TS}.log"

# -----------------------------
# 解析参数：--smoke / --data
# -----------------------------
SMOKE=0
DATA_ARG=""
if [ "${1:-}" = "--smoke" ]; then
  SMOKE=1
  shift
fi
if [ "${1:-}" = "--data" ]; then
  DATA_ARG="${2:-}"
  shift 2 || true
fi

# -----------------------------
# 定位本地模型（不触网）
# -----------------------------
echo "[INFO] Resolving local model path..." | tee -a "$LOG"
RESOLVE_OUT="$("$PY" -u "$SCRIPT_DIR/main.py" \
  --repo-id "$MODEL_ID" \
  --local-dir "$LOCAL_DIR" \
  --cache-dir "$CACHE_DIR" \
  --fallback-repo "$FALLBACK_REPO" \
  --download-only 2>&1 | tee -a "$LOG")"

MODEL_PATH="$(echo "$RESOLVE_OUT" | sed -n 's/^\[OK\] Model ready at: //p' | tail -n 1)"
if [ -z "${MODEL_PATH:-}" ]; then
  echo "[FATAL] Failed to parse model path from main.py output. See log: $LOG" >&2
  echo "        常见原因：GPU 机没有本地模型（CPU 机没提前下载）" >&2
  exit 2
fi
export CED_MODEL_PATH="$MODEL_PATH"
echo "[OK] CED_MODEL_PATH=$CED_MODEL_PATH" | tee -a "$LOG"

# -----------------------------
# Smoke test：加载 + 短生成
# -----------------------------
if [ "$SMOKE" -eq 1 ]; then
  echo "[INFO] Running smoke test (load + short generation) ..." | tee -a "$LOG"
  "$PY" -u "$SCRIPT_DIR/main.py" \
    --repo-id "$MODEL_ID" \
    --local-dir "$LOCAL_DIR" \
    --cache-dir "$CACHE_DIR" \
    --fallback-repo "$FALLBACK_REPO" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --smoke-test 2>&1 | tee -a "$LOG"
  echo "[DONE] smoke test ok. Log: $LOG"
  exit 0
fi

# -----------------------------
# 自定义命令模式：bash gpu.sh python xxx.py ...
# -----------------------------
if [ "$#" -gt 0 ]; then
  cmd=( "$@" )
  for i in "${!cmd[@]}"; do
    cmd[$i]="${cmd[$i]//\{MODEL_PATH\}/$CED_MODEL_PATH}"
  done
  echo "[INFO] Running user command:" | tee -a "$LOG"
  echo "       ${cmd[*]}" | tee -a "$LOG"
  "${cmd[@]}" 2>&1 | tee -a "$LOG"
  echo "[DONE] user cmd ok. Log: $LOG"
  exit 0
fi

# -----------------------------
# 默认：跑“整个数据集”
# -----------------------------
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/logs/dataset_run_${TS}}"
mkdir -p "$RUN_DIR"
OUT_PRED="${OUT_PRED:-$RUN_DIR/predictions.jsonl}"
OUT_SUMMARY="${OUT_SUMMARY:-$RUN_DIR/summary.json}"

DATA_PATH="${DATA_PATH:-$DATA_ARG}"

echo "[INFO] Default: run full dataset" | tee -a "$LOG"
echo "[INFO] RUN_DIR=$RUN_DIR" | tee -a "$LOG"
echo "[INFO] OUT_PRED=$OUT_PRED" | tee -a "$LOG"
echo "[INFO] OUT_SUMMARY=$OUT_SUMMARY" | tee -a "$LOG"
if [ -n "${DATA_PATH:-}" ]; then
  echo "[INFO] DATA_PATH (explicit) = $DATA_PATH" | tee -a "$LOG"
else
  echo "[INFO] DATA_PATH not provided. Will auto-discover dataset under ../dataprepare/ ..." | tee -a "$LOG"
fi

"$PY" -u "$SCRIPT_DIR/run_full_dataset.py" \
  --model_path "$CED_MODEL_PATH" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  ${DATA_PATH:+ --data "$DATA_PATH"} \
  --out_pred "$OUT_PRED" \
  --out_summary "$OUT_SUMMARY" \
  2>&1 | tee -a "$LOG"

echo "[DONE] dataset run ok."
echo "       summary : $OUT_SUMMARY"
echo "       preds   : $OUT_PRED"
echo "       log     : $LOG"
