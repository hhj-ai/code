#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# gpu_run.sh
# GPU服务器（无网，8×H200）运行：安装环境 + P0-a/P0-b 实验
#
# 路径与原 gpu.sh 完全对齐：
#   CED_ENV    = .../hhj-train/dataprepare/conda_envs/ced_p0
#   MODEL_DIR  = $PWD/../dataprepare/models/Qwen3-VL-8B-Instruct
#   COCO_DIR   = $PWD/../dataprepare/data/coco
#   VQA_DIR    = $PWD/../dataprepare/data/ced_vqa
#   PIP_CACHE  = $PWD/../dataprepare/pip_wheels
#
# 用法：
#   bash gpu_run.sh               # 完整流程：P0-a → P0-b → 分析
#   bash gpu_run.sh --probe-only  # 仅P0-a架构探测
#   bash gpu_run.sh --skip-probe  # 跳过P0-a，直接跑P0-b
#   bash gpu_run.sh --analysis    # 仅对已有结果跑分析
#   bash gpu_run.sh --smoke       # 只验活（加载+短生成，与原gpu.sh兼容）
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 路径配置（与原 gpu.sh 完全对齐） ----------

CED_ENV="${CED_ENV:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare/conda_envs/ced_p0}"
PY="${PYTHON_BIN:-$CED_ENV/bin/python}"
PIP="$CED_ENV/bin/pip"

if [ ! -x "$PY" ]; then
  echo "[FATAL] Python not found: $PY" >&2
  echo "        请先在 CPU 服务器运行 cpu_download.sh" >&2
  exit 1
fi

# ---------- LD_LIBRARY_PATH（与原 gpu.sh 完全一致） ----------

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

# ---------- HF 缓存（与原 gpu.sh 完全对齐） ----------

export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# GPU 无网：强制离线（与原 gpu.sh 一致）
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# ---------- 数据路径（与原 gpu.sh 的 dataprepare 结构对齐） ----------

MODEL_DIR="${LOCAL_DIR:-$PWD/../dataprepare/models/Qwen3-VL-8B-Instruct}"
COCO_DIR="$PWD/../dataprepare/data/coco"
VQA_DIR="$PWD/../dataprepare/data/ced_vqa"
PIP_CACHE="$PWD/../dataprepare/pip_wheels"
RESULTS_DIR="$PWD/results"

DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"

# ---------- 日志（与原 gpu.sh 的 logs 目录一致） ----------

mkdir -p "$PWD/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="$PWD/logs/gpu_${TS}.log"

# ---------- 参数解析 ----------

MODE="full"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --probe-only)  MODE="probe"; shift;;
    --skip-probe)  MODE="validate"; shift;;
    --analysis)    MODE="analysis"; shift;;
    --smoke)       MODE="smoke"; shift;;
    --device)      DEVICE="$2"; shift 2;;
    --dtype)       DTYPE="$2"; shift 2;;
    *)             echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "============================================" | tee -a "$LOG"
echo "  CED P0 - GPU Experiment Runner"            | tee -a "$LOG"
echo "  MODE      : $MODE"                         | tee -a "$LOG"
echo "  CED_ENV   : $CED_ENV"                      | tee -a "$LOG"
echo "  MODEL_DIR : $MODEL_DIR"                    | tee -a "$LOG"
echo "  COCO_DIR  : $COCO_DIR"                     | tee -a "$LOG"
echo "  VQA_DIR   : $VQA_DIR"                      | tee -a "$LOG"
echo "  DEVICE    : $DEVICE"                       | tee -a "$LOG"
echo "  RESULTS   : $RESULTS_DIR"                  | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

# ---------- 离线安装缺失包（从共享目录的 wheel 缓存） ----------

echo "[ENV] Checking/installing packages offline ..." | tee -a "$LOG"
if [ -d "$PIP_CACHE" ]; then
    "$PIP" install --no-index --find-links="$PIP_CACHE" \
        -r "$SCRIPT_DIR/requirements.txt" 2>&1 | tee -a "$LOG" \
        || echo "[WARN] Some packages may already be installed or unavailable offline."
fi

# 验证关键导入
"$PY" -c "
import torch, transformers, PIL, numpy, scipy, sklearn
print(f'[OK] torch={torch.__version__} transformers={transformers.__version__}')
print(f'[OK] CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}')
" 2>&1 | tee -a "$LOG" || { echo "[FATAL] Import check failed" | tee -a "$LOG"; exit 1; }

# ---------- 检查数据 ----------

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "[FATAL] Model not found at $MODEL_DIR" | tee -a "$LOG"
    echo "        请先在 CPU 服务器运行 cpu_download.sh" | tee -a "$LOG"
    exit 1
fi

VQA_FILE="$VQA_DIR/ced_vqa_dataset.jsonl"
COCO_IMAGE_DIR="$COCO_DIR/val2017"

# smoke 模式不需要 VQA 数据
if [ "$MODE" != "smoke" ] && [ "$MODE" != "analysis" ]; then
    if [ ! -f "$VQA_FILE" ]; then
        echo "[FATAL] VQA dataset not found: $VQA_FILE" | tee -a "$LOG"
        echo "        请先在 CPU 服务器运行 cpu_download.sh" | tee -a "$LOG"
        exit 1
    fi
    if [ ! -d "$COCO_IMAGE_DIR" ]; then
        echo "[FATAL] COCO images not found: $COCO_IMAGE_DIR" | tee -a "$LOG"
        exit 1
    fi
fi

mkdir -p "$RESULTS_DIR/figures"

# ============================================================
# Smoke test（与原 gpu.sh --smoke 兼容）
# ============================================================
if [ "$MODE" = "smoke" ]; then
    echo "[INFO] Running smoke test ..." | tee -a "$LOG"
    "$PY" -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/src')
from model_loader import load_qwen3vl, get_num_layers
import torch

processor, model, cfg = load_qwen3vl('$MODEL_DIR', '$DEVICE', '$DTYPE')
n = get_num_layers(cfg, model)
print(f'[OK] Loaded model. num_layers={n}')

messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Hello World'}]}]
inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
inputs.pop('token_type_ids', None)
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.no_grad():
    gen = model.generate(**inputs, max_new_tokens=16)
out = processor.batch_decode(gen[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
print(f'[SMOKE_TEST_OUTPUT] {out}')
" 2>&1 | tee -a "$LOG"
    echo "[DONE] smoke test ok. Log: $LOG"
    exit 0
fi

# ============================================================
# P0-a: 架构探测
# ============================================================
run_probe() {
    echo ""                                             | tee -a "$LOG"
    echo "========== P0-a: Architecture Probe ==========" | tee -a "$LOG"
    "$PY" -u "$SCRIPT_DIR/src/p0a_probe.py" \
        --model_dir "$MODEL_DIR" \
        --coco_dir "$COCO_DIR" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --output "$RESULTS_DIR/p0a_probe_report.json" \
        2>&1 | tee -a "$LOG"
    echo "[DONE] P0-a report: $RESULTS_DIR/p0a_probe_report.json" | tee -a "$LOG"
}

# ============================================================
# P0-b: 完整 CED 验证
# ============================================================
run_validate() {
    echo ""                                              | tee -a "$LOG"
    echo "========== P0-b: CED Validation ==========="   | tee -a "$LOG"
    "$PY" -u "$SCRIPT_DIR/src/p0b_validate.py" \
        --model_dir "$MODEL_DIR" \
        --vqa_file "$VQA_FILE" \
        --coco_image_dir "$COCO_IMAGE_DIR" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --output_raw "$RESULTS_DIR/p0b_raw.jsonl" \
        --output_summary "$RESULTS_DIR/p0b_summary.json" \
        --layers "logits,16,20,24,28,32" \
        --lambda_e_sweep "0.0,0.05,0.1,0.2,0.3,0.5" \
        2>&1 | tee -a "$LOG"
    echo "[DONE] P0-b raw: $RESULTS_DIR/p0b_raw.jsonl"    | tee -a "$LOG"
}

# ============================================================
# 分析
# ============================================================
run_analysis() {
    echo ""                                           | tee -a "$LOG"
    echo "========== Analysis ========================" | tee -a "$LOG"

    if [ ! -f "$RESULTS_DIR/p0b_raw.jsonl" ]; then
        echo "[FATAL] p0b_raw.jsonl not found, run P0-b first" | tee -a "$LOG"
        exit 1
    fi

    "$PY" -u "$SCRIPT_DIR/src/analysis.py" \
        --raw_file "$RESULTS_DIR/p0b_raw.jsonl" \
        --output_dir "$RESULTS_DIR" \
        2>&1 | tee -a "$LOG"
    echo "[DONE] Analysis outputs in $RESULTS_DIR/"   | tee -a "$LOG"
}

# ============================================================
# 按模式执行
# ============================================================
case "$MODE" in
    probe)
        run_probe
        ;;
    validate)
        run_validate
        run_analysis
        ;;
    analysis)
        run_analysis
        ;;
    full)
        run_probe
        run_validate
        run_analysis
        ;;
esac

echo ""                                                    | tee -a "$LOG"
echo "============================================"        | tee -a "$LOG"
echo "  All done. Log: $LOG"                               | tee -a "$LOG"
echo "  Results:  $RESULTS_DIR/"                           | tee -a "$LOG"
echo "============================================"        | tee -a "$LOG"
