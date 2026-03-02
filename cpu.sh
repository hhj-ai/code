#!/bin/bash
# ============================================================================
# cpu.sh (CPU has internet) - Qwen3-VL + Transformers (PyPI) Offline Prep
# ============================================================================
# What this script does:
#  1) Create/activate shared conda env (prefix) on shared filesystem
#  2) FORCE pip to use OFFICIAL PyPI (ignores any internal pip.conf)
#  3) Download wheels into shared wheelhouse for OFFLINE GPU install
#     - transformers==5.1.0 (supports qwen3_vl)
#     - huggingface_hub/accelerate/tokenizers/safetensors/etc
#     - torch/torchvision from official PyTorch CUDA index
#  4) Download Qwen3-VL model snapshot into shared models dir (resume, no symlinks)
#  5) (Optional) COCO presence check (does not download COCO)
#
# No self-copying, no renaming, no writing into your git repo.
# ============================================================================

set -euo pipefail

# ----------------------------
# Shared paths
# ----------------------------
BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare"
CODE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/code"
ENV_DIR="${BASE_DIR}/conda_envs/ced_p0"
WHEELHOUSE="${BASE_DIR}/packages"
HF_HOME_DIR="${BASE_DIR}/.hf_cache"

# ----------------------------
# Model config
# ----------------------------
export MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-VL-8B-Instruct}"
export MODEL_DIR="${BASE_DIR}/models/$(basename "${MODEL_REPO}")"

# ----------------------------
# FORCE OFFICIAL PyPI
# ----------------------------
export PIP_CONFIG_FILE="/dev/null"
export PIP_INDEX_URL="https://pypi.org/simple"
export PIP_TRUSTED_HOST="pypi.org"
export PIP_EXTRA_INDEX_URL=""
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

PYPI="https://pypi.org/simple"

# ----------------------------
# HF endpoint (mirror optional)
# ----------------------------
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME_OVERRIDE:-$HF_HOME_DIR}"

# ----------------------------
# COCO check paths
# ----------------------------
COCO_DIR="${BASE_DIR}/data/coco"
COCO_IMG_DIR="${COCO_DIR}/images/val2017"
COCO_ANN_FILE="${COCO_DIR}/annotations/instances_val2017.json"

echo "============================================"
echo "[CPU] BASE_DIR    : ${BASE_DIR}"
echo "[CPU] CODE_DIR    : ${CODE_DIR}"
echo "[CPU] ENV_DIR     : ${ENV_DIR}"
echo "[CPU] WHEELHOUSE  : ${WHEELHOUSE}"
echo "[CPU] HF_ENDPOINT : ${HF_ENDPOINT}"
echo "[CPU] HF_HOME     : ${HF_HOME}"
echo "[CPU] MODEL_REPO  : ${MODEL_REPO}"
echo "[CPU] MODEL_DIR   : ${MODEL_DIR}"
echo "============================================"

mkdir -p "${WHEELHOUSE}" "${HF_HOME}" "${MODEL_DIR}" "${CODE_DIR}"
mkdir -p "${COCO_DIR}/images" "${COCO_DIR}/annotations"

# -------------------------------------------------------------------
# Step 1: Create/activate conda env (shared prefix)
# -------------------------------------------------------------------
echo "[1/5] Creating/activating conda environment..."

__conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null || true)"
if [ -n "${__conda_setup}" ]; then
  eval "$__conda_setup"
else
  CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

if [ -d "${ENV_DIR}" ] && [ -f "${ENV_DIR}/bin/python" ]; then
  echo "[CPU] conda env exists: ${ENV_DIR}"
else
  echo "[CPU] creating conda env: ${ENV_DIR}"
  conda create --prefix "${ENV_DIR}" --override-channels -c defaults python=3.10.0 -y
fi

conda activate "${ENV_DIR}"
echo "[CPU] python: $(python -V) @ $(which python)"

python -m pip install --upgrade pip setuptools wheel -i "${PYPI}" --trusted-host pypi.org

# -------------------------------------------------------------------
# Step 2: Download wheels
# -------------------------------------------------------------------
echo "[2/5] Downloading wheels to wheelhouse..."

# Transformers (new enough for qwen3_vl)
python -m pip download -d "${WHEELHOUSE}" -i "${PYPI}" --trusted-host pypi.org   "transformers==5.1.0"

# HuggingFace + runtime deps
python -m pip download -d "${WHEELHOUSE}" -i "${PYPI}" --trusted-host pypi.org   "huggingface_hub[cli]==0.28.1"   "accelerate==1.3.0"   "safetensors==0.5.2"   "tokenizers==0.21.0"   "sentencepiece==0.2.0"   "qwen-vl-utils==0.0.10"   "numpy==1.26.4"   "scipy==1.14.1"   "scikit-learn==1.6.1"   "matplotlib==3.9.4"   "seaborn==0.13.2"   "Pillow==11.1.0"   "pycocotools==2.0.8"   "tqdm==4.67.1"   "pandas==2.2.3"   "av==14.1.0"

# Torch CUDA wheels (official PyTorch index)
TORCH_IDX="https://download.pytorch.org/whl/cu124"
python -m pip download -d "${WHEELHOUSE}" --index-url "${TORCH_IDX}"   torch==2.5.1 torchvision==0.20.1

echo "[CPU] wheelhouse file count:"
ls -1 "${WHEELHOUSE}" | wc -l | awk '{print "  files:", $1}'
ls -lh "${WHEELHOUSE}"/transformers-5.1.0*.whl 2>/dev/null || true

# -------------------------------------------------------------------
# Step 3: Download Qwen3-VL model snapshot
# -------------------------------------------------------------------
echo "[3/5] Downloading/resuming model snapshot to ${MODEL_DIR} ..."

python -m pip install -U "huggingface_hub[cli]==0.28.1" -i "${PYPI}" --trusted-host pypi.org

python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ.get("MODEL_REPO")
local_dir = os.environ.get("MODEL_DIR")
if not repo_id or not local_dir:
    raise RuntimeError(f"MODEL_REPO/MODEL_DIR not set: MODEL_REPO={repo_id}, MODEL_DIR={local_dir}")

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"[CPU] snapshot_download done: {repo_id} -> {local_dir}")
PY

echo "[CPU] sanity check model dir (top 30):"
ls -lh "${MODEL_DIR}" | head -n 30

# -------------------------------------------------------------------
# Step 4: COCO check (no download)
# -------------------------------------------------------------------
echo "[4/5] COCO check..."
if [ -d "${COCO_IMG_DIR}" ] && [ "$(ls -1 "${COCO_IMG_DIR}" 2>/dev/null | wc -l)" -gt 4000 ] && [ -f "${COCO_ANN_FILE}" ]; then
  echo "[CPU] COCO val2017 OK: ${COCO_IMG_DIR} + ${COCO_ANN_FILE}"
else
  echo "[CPU] COCO not complete (skipping download)."
  echo "      images: ${COCO_IMG_DIR}"
  echo "      ann   : ${COCO_ANN_FILE}"
fi

# -------------------------------------------------------------------
# Step 5: Done
# -------------------------------------------------------------------
echo "[5/5] Done."
echo "============================================"
echo "[CPU] READY."
echo "Wheelhouse : ${WHEELHOUSE}"
echo "Model dir  : ${MODEL_DIR}"
echo "Next (GPU): run gpu_official.sh"
echo "============================================"
