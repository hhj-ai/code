#!/bin/bash
# ============================================================================
# CPU Server Download Script (FORCE OFFICIAL PYPI) for CED Phase 0 / Qwen3-VL
# - CPU server has internet
# - Downloads wheels into shared wheelhouse for offline GPU install
# - Downloads model weights via HF mirror endpoint (configurable)
#
# Key goals:
#   1) Force pip to use OFFICIAL PyPI (https://pypi.org/simple) regardless of
#      any system/company pip.conf (by setting PIP_CONFIG_FILE and env vars,
#      and always passing -i).
#   2) Build a Transformers *git* wheel without dependency resolution
#      (pip wheel --no-deps) to avoid private-index dependency issues.
#   3) Download Qwen3-VL model snapshots into shared model dir.
# ============================================================================

set -euo pipefail

# ----------------------------
# Paths (shared filesystem)
# ----------------------------
BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare"
CODE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/code"
ENV_DIR="${BASE_DIR}/conda_envs/ced_p0"
PKG_DIR="${BASE_DIR}/packages"
HF_CACHE_DIR="${BASE_DIR}/.hf_cache"

# ----------------------------
# Model selection (Qwen3-VL)
# ----------------------------
# Default to 8B to keep download/runtime reasonable; override if you want.
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_DIR="${BASE_DIR}/models/$(basename "${MODEL_REPO}")"

# ----------------------------
# Force OFFICIAL PyPI
# ----------------------------
export PIP_INDEX_URL="https://pypi.org/simple"
export PIP_TRUSTED_HOST="pypi.org"
export PIP_EXTRA_INDEX_URL=""
export PIP_NO_INPUT=1
# Ignore any global/user/company pip.conf that might point to internal indexes
export PIP_CONFIG_FILE="/dev/null"

PYPI="https://pypi.org/simple"

# ----------------------------
# HF endpoint (mirror optional)
# ----------------------------
# You asked for mirrors earlier; keep this configurable.
# Set HF_ENDPOINT=https://huggingface.co if you want official HF.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$HF_CACHE_DIR}"

echo "============================================"
echo "CED CPU download script (FORCE OFFICIAL PYPI)"
echo "BASE_DIR   : ${BASE_DIR}"
echo "CODE_DIR   : ${CODE_DIR}"
echo "ENV_DIR    : ${ENV_DIR}"
echo "PKG_DIR    : ${PKG_DIR}"
echo "HF_HOME    : ${HF_HOME}"
echo "HF_ENDPOINT: ${HF_ENDPOINT}"
echo "MODEL_REPO : ${MODEL_REPO}"
echo "MODEL_DIR  : ${MODEL_DIR}"
echo "============================================"

mkdir -p "${PKG_DIR}" "${CODE_DIR}" "${HF_HOME}" "${MODEL_DIR}"

# -------------------------------------------------------------------
# Step 1: Create/activate conda environment (shared prefix)
# -------------------------------------------------------------------
echo "[1/5] Creating/activating conda environment..."

__conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null || true)"
if [ -n "${__conda_setup}" ]; then
  eval "$__conda_setup"
else
  CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
  if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
  else
    echo "ERROR: Cannot initialize conda."
    exit 1
  fi
fi

if [ -d "${ENV_DIR}" ] && [ -f "${ENV_DIR}/bin/python" ]; then
  echo "Conda env already exists at ${ENV_DIR}, skipping creation."
else
  conda create --prefix "${ENV_DIR}" --override-channels -c defaults python=3.10.0 -y
fi

conda activate "${ENV_DIR}"
echo "Activated python: $(python -V) @ $(which python)"

# Upgrade build tooling strictly from official PyPI
python -m pip install --upgrade pip setuptools wheel -i "${PYPI}" --trusted-host pypi.org

# -------------------------------------------------------------------
# Step 2: Download wheels for core deps (offline GPU install)
# -------------------------------------------------------------------
echo "[2/5] Downloading wheelhouse from OFFICIAL PyPI..."

# CUDA PyTorch wheels: keep as you had (from PyTorch official index).
TORCH_IDX="https://download.pytorch.org/whl/cu124"
python -m pip download -d "${PKG_DIR}" --index-url "${TORCH_IDX}" \
  torch==2.5.1 torchvision==0.20.1

# Hugging Face hub + CLI (needed for snapshot_download and hf CLI)
python -m pip download -d "${PKG_DIR}" -i "${PYPI}" --trusted-host pypi.org \
  "huggingface_hub[cli]==0.28.1" \
  "safetensors==0.5.2" \
  "tokenizers==0.21.0" \
  "sentencepiece==0.2.0" \
  "accelerate==1.3.0" \
  "qwen-vl-utils==0.0.10" \
  "numpy==1.26.4" \
  "scipy==1.14.1" \
  "scikit-learn==1.6.1" \
  "matplotlib==3.9.4" \
  "seaborn==0.13.2" \
  "Pillow==11.1.0" \
  "pycocotools==2.0.8" \
  "tqdm==4.67.1" \
  "pandas==2.2.3" \
  "av==14.1.0"

# -------------------------------------------------------------------
# Step 3: Build 최신 Transformers from git (wheel only, NO DEPS)
#   Qwen3-VL model cards recommend installing transformers from source.
#   We avoid dependency resolution here to dodge any private-index issues.
# -------------------------------------------------------------------
echo "[3/5] Building transformers wheel from git (no dependency resolution)..."

python -m pip wheel --no-deps -w "${PKG_DIR}" \
  "git+https://github.com/huggingface/transformers.git"

# Also download runtime deps that Transformers typically needs (from official PyPI)
python -m pip download -d "${PKG_DIR}" -i "${PYPI}" --trusted-host pypi.org \
  "requests>=2.26.0" "packaging>=20.0" "pyyaml>=5.1" "regex!=2019.12.17" \
  "filelock" "fsspec" "typing-extensions" "jinja2" "sympy" "networkx"

# -------------------------------------------------------------------
# Step 4: Download model snapshot to shared dir (using HF_ENDPOINT mirror)
#   Use Python snapshot_download so we can enforce local_dir_use_symlinks=False.
# -------------------------------------------------------------------
echo "[4/5] Downloading/resuming model snapshot..."

# Install huggingface_hub locally (from official PyPI) so Python API is available
python -m pip install --upgrade -i "${PYPI}" --trusted-host pypi.org "huggingface_hub[cli]==0.28.1"

python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_REPO"]
local_dir = os.environ["MODEL_DIR"]

# This will resume automatically when possible.
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"[MODEL] snapshot_download done: {repo_id} -> {local_dir}")
PY

# Basic completeness check: must have some weight file
if ! ls -1 "${MODEL_DIR}"/*.safetensors >/dev/null 2>&1; then
  echo "ERROR: No .safetensors found in ${MODEL_DIR} after download."
  echo "       Check HF_ENDPOINT / network."
  exit 1
fi

# -------------------------------------------------------------------
# Step 5: Copy code (NO renaming, NO self-copy)
# -------------------------------------------------------------------
echo "[5/5] Copying code artifacts (no renaming)..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for f in main.py gpu.sh; do
  if [ -f "${SCRIPT_DIR}/${f}" ]; then
    cp "${SCRIPT_DIR}/${f}" "${CODE_DIR}/${f}"
    echo "  Copied ${f} -> ${CODE_DIR}/"
  fi
done

echo ""
echo "============================================"
echo "CPU stage complete (OFFICIAL PYPI enforced)."
echo "Wheelhouse : ${PKG_DIR}"
echo "Model dir  : ${MODEL_DIR}"
echo "HF cache   : ${HF_HOME}"
echo "Next on GPU: cd ${CODE_DIR} && bash gpu.sh"
echo "============================================"
