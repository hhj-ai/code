#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# gpu.sh (v12)
# 和 cpu.sh 对齐：
# - 同一个 CED_ENV / HF_HOME / CACHE 变量逻辑
# - 先用 main.py --download-only 解析本地模型路径（不触网）
# - 默认跑“整数据集”，但只在 dataprepare/data 与 dataprepare/datasets 下自动发现
#   （避免误选 conda env / site-packages 里的乱七八糟 json）
#
# 用法：
#   bash gpu.sh                      # 默认：自动发现数据集 + 全量跑
#   bash gpu.sh --data /path/a.jsonl # 指定数据集
#   bash gpu.sh --smoke              # 只验活（加载+短生成）
#   bash gpu.sh --exec -- <cmd ...>  # 执行自定义命令（支持 {MODEL_PATH} 占位符）
# ------------------------------------------------------------

CED_ENV="${CED_ENV:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare/conda_envs/ced_p0}"
PY="${PYTHON_BIN:-$CED_ENV/bin/python}"
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

# ✅ 与 cpu.sh 对齐：HF_HOME 默认用当前目录
export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# GPU 无网：强制离线（避免卡住）
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

MODEL_ID="${MODEL_ID:-auto}"
LOCAL_DIR="${LOCAL_DIR:-$PWD/../dataprepare/models/Qwen3-VL-8B-Instruct}"
CACHE_DIR="${CACHE_DIR:-$HF_HUB_CACHE}"
FALLBACK_REPO="${FALLBACK_REPO:-Qwen/Qwen3-VL-8B-Instruct}"

DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
TEMPERATURE="${TEMPERATURE:-0.0}"

# 日志目录
mkdir -p "$PWD/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$PWD/logs/gpu_${TS}.log"

# -----------------------------
# 解析本地模型路径（不加载权重）
# -----------------------------
echo "[INFO] Resolving local model path..." | tee -a "$LOG"
RESOLVE_OUT="$("$PY" -u main.py \
  --repo-id "$MODEL_ID" \
  --local-dir "$LOCAL_DIR" \
  --cache-dir "$CACHE_DIR" \
  --fallback-repo "$FALLBACK_REPO" \
  --download-only 2>&1 | tee -a "$LOG")"

MODEL_PATH="$(echo "$RESOLVE_OUT" | sed -n 's/^\[OK\] Model ready at: //p' | tail -n 1)"
if [ -z "${MODEL_PATH:-}" ]; then
  echo "[FATAL] Failed to parse model path from main.py output. See: $LOG" >&2
  exit 2
fi
export CED_MODEL_PATH="$MODEL_PATH"
echo "[OK] CED_MODEL_PATH=$CED_MODEL_PATH" | tee -a "$LOG"

# -----------------------------
# 参数解析
# -----------------------------
DATA_PATH="${DATA_PATH:-}"
MODE="full"

# 允许：gpu.sh --data xxx
while [ "$#" -gt 0 ]; do
  case "$1" in
    --data)
      DATA_PATH="$2"; shift 2;;
    --smoke)
      MODE="smoke"; shift;;
    --exec)
      MODE="exec"; shift; break;;
    --)
      shift; break;;
    *)
      # 兼容：用户直接传命令 => exec 模式
      MODE="exec"; break;;
  esac
done

if [ "$MODE" = "smoke" ]; then
  echo "[INFO] Running smoke test (load + short generation)..." | tee -a "$LOG"
  "$PY" -u main.py \
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

if [ "$MODE" = "exec" ]; then
  # 自定义命令（支持 {MODEL_PATH} 占位符）
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
# 默认：跑整数据集
# -----------------------------
RUN_DIR="$PWD/logs/dataset_run_${TS}"
mkdir -p "$RUN_DIR"
OUT_PRED="$RUN_DIR/predictions.jsonl"
OUT_SUM="$RUN_DIR/summary.json"

# 默认只在这两个目录里自动发现，避免误选 conda env/site-packages
DEFAULT_DATA_ROOTS=(
  "$PWD/../dataprepare/data"
  "$PWD/../dataprepare/datasets"
)

ARGS=( "--model_path" "$CED_MODEL_PATH"
       "--device" "$DEVICE"
       "--dtype" "$DTYPE"
       "--max_new_tokens" "$MAX_NEW_TOKENS"
       "--temperature" "$TEMPERATURE"
       "--out_pred" "$OUT_PRED"
       "--out_summary" "$OUT_SUM" )

if [ -n "$DATA_PATH" ]; then
  ARGS+=( "--data" "$DATA_PATH" )
else
  # 传给脚本：只搜索这些 roots
  for r in "${DEFAULT_DATA_ROOTS[@]}"; do
    ARGS+=( "--data_root" "$r" )
  done
fi

echo "[INFO] Running full dataset evaluation..." | tee -a "$LOG"
"$PY" -u run_full_dataset.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"

echo "[DONE] dataset run ok."
echo "       summary : $OUT_SUM"
echo "       preds   : $OUT_PRED"
echo "       log     : $LOG"
