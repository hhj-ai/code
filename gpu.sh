#!/bin/bash
# =============================================================================
# gpu.sh (GPU server has NO internet)
#
# Goal: Offline install a compatible stack (Transformers 5.2.0 + hub 1.4.1
# + tokenizers 0.22.2) and run your CED Phase 0 main.py.
#
# Fixes your error:
#   ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
# which happens when huggingface-hub is too old (e.g., 0.28.1).
#
# Also keeps the CUDA nvJitLink workaround to avoid symbol mismatch issues.
# =============================================================================

set -euo pipefail

BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare"
CODE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/code"
ENV_DIR="${BASE_DIR}/conda_envs/ced_p0"
WHEELHOUSE="${BASE_DIR}/packages"

export MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-VL-8B-Instruct}"
export MODEL_DIR="${BASE_DIR}/models/$(basename "${MODEL_REPO}")"

COCO_IMG_DIR="${BASE_DIR}/data/coco/images/val2017"
COCO_ANN_FILE="${BASE_DIR}/data/coco/annotations/instances_val2017.json"
OUT_DIR="${CODE_DIR}/results_p0"

echo "============================================================"
echo "[GPU] ENV_DIR    : ${ENV_DIR}"
echo "[GPU] WHEELHOUSE : ${WHEELHOUSE}"
echo "[GPU] MODEL_DIR  : ${MODEL_DIR}"
echo "============================================================"

# -------------------------------------------------------------------
# Step 1: Activate conda prefix env
# -------------------------------------------------------------------
__conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null || true)"
if [ -n "${__conda_setup}" ]; then
  eval "$__conda_setup"
else
  CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

# Clean nested activations (common source of weirdness)
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate "${ENV_DIR}"

echo "[GPU] CONDA_PREFIX=${CONDA_PREFIX}"
echo "[GPU] python=$(which python)"
python -V

# -------------------------------------------------------------------
# Step 2: CUDA nvJitLink fix (prefer wheel-bundled nvjitlink over /usr/local/cuda)
# -------------------------------------------------------------------
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"

# -------------------------------------------------------------------
# Step 3: Offline flags
#   - huggingface_hub offline mode uses HF_HUB_OFFLINE=1
# -------------------------------------------------------------------
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# -------------------------------------------------------------------
# Step 4: Offline install (FORCE correct versions)
# -------------------------------------------------------------------
echo "[GPU] Uninstall conflicting old packages (if any)..."
pip uninstall -y transformers huggingface-hub tokenizers || true

echo "[GPU] Install torch/torchvision from wheelhouse (offline)..."
pip install --no-index --find-links="${WHEELHOUSE}" torch==2.5.1 torchvision==0.20.1 || true

echo "[GPU] Install pinned compatible HF stack (offline)..."
pip install --no-index --find-links="${WHEELHOUSE}" \
  tokenizers==0.22.2 \
  huggingface-hub==1.4.1 \
  transformers==5.2.0

echo "[GPU] Install remaining pinned deps (offline)..."
REQ_FILE="${WHEELHOUSE}/requirements.offline.txt"
if [ ! -f "${REQ_FILE}" ]; then
  echo "ERROR: ${REQ_FILE} not found. Run cpu.sh first."
  exit 1
fi
# Install everything else from the lockfile (will be no-op for already-installed pins)
pip install --no-index --find-links="${WHEELHOUSE}" -r "${REQ_FILE}"

echo "[GPU] Verify imports & versions..."
python - <<'PY'
import torch, transformers
from huggingface_hub import is_offline_mode
print("PyTorch      :", torch.__version__, "CUDA=", torch.cuda.is_available(), "GPUs=", torch.cuda.device_count())
print("Transformers :", transformers.__version__)
print("HF hub offline:", is_offline_mode())
PY

echo "[GPU] Verify qwen3_vl config is recognized..."
python - <<'PY'
import os
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(os.environ["MODEL_DIR"], trust_remote_code=True)
print("config.model_type =", getattr(cfg, "model_type", None))
PY

# -------------------------------------------------------------------
# Step 5: Run experiments
# -------------------------------------------------------------------
mkdir -p "${OUT_DIR}"
cd "${CODE_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"

echo ""
echo "========== P0-a: Architecture Probing =========="
python main.py \
  --mode probe \
  --model_path "${MODEL_DIR}" \
  --coco_img_dir "${COCO_IMG_DIR}" \
  --coco_ann_file "${COCO_ANN_FILE}" \
  --output_dir "${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}/log_p0a.txt"

echo ""
echo "========== P0-b: CED Signal Validation =========="
python main.py \
  --mode validate \
  --model_path "${MODEL_DIR}" \
  --coco_img_dir "${COCO_IMG_DIR}" \
  --coco_ann_file "${COCO_ANN_FILE}" \
  --output_dir "${OUT_DIR}" \
  --num_samples 500 \
  --layers logits 12 16 20 24 27 \
  --lambda_e_values 0.0 0.05 0.1 0.2 0.3 0.5 \
  2>&1 | tee "${OUT_DIR}/log_p0b.txt"

echo "============================================================"
echo "[GPU] DONE. Results: ${OUT_DIR}"
echo "============================================================"
