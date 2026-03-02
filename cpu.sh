#!/bin/bash
# =============================================================================
# cpu.sh (CPU server HAS internet)
#
# Goal: Prepare an OFFLINE-friendly wheelhouse + download Qwen3-VL model files.
# Fixes all dependency conflicts you saw by aligning versions with
# Transformers 5.x requirements (tokenizers>=0.22, huggingface-hub>=1.3).
#
# Key references:
# - Latest Transformers release on PyPI is 5.2.0 (Feb 16, 2026). 
# - Latest huggingface-hub release on PyPI is 1.4.1 (Feb 6, 2026).
# - Latest tokenizers release line is 0.22.x (e.g., 0.22.2 Jan 5, 2026).
# - huggingface_hub provides is_offline_mode() and uses HF_HUB_OFFLINE=1.
#
# This script:
#  1) Creates/activates shared conda env (prefix) on shared filesystem
#  2) Forces pip to use OFFICIAL PyPI (ignores any internal pip.conf)
#  3) Downloads wheels into shared wheelhouse (for GPU offline install)
#  4) Downloads Qwen3-VL model snapshot into shared models dir
#
# No self-copying / no renaming / no git operations.
# =============================================================================

set -euo pipefail

# ----------------------------
# Shared paths
# ----------------------------
BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare"
ENV_DIR="${BASE_DIR}/conda_envs/ced_p0"
WHEELHOUSE="${BASE_DIR}/packages"
HF_HOME_DIR="${BASE_DIR}/.hf_cache"

# ----------------------------
# Model selection
# ----------------------------
export MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-VL-8B-Instruct}"
export MODEL_DIR="${BASE_DIR}/models/$(basename "${MODEL_REPO}")"

# ----------------------------
# Force OFFICIAL PyPI
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

mkdir -p "${WHEELHOUSE}" "${HF_HOME}" "${MODEL_DIR}"

echo "============================================================"
echo "[CPU] ENV_DIR     : ${ENV_DIR}"
echo "[CPU] WHEELHOUSE  : ${WHEELHOUSE}"
echo "[CPU] HF_ENDPOINT : ${HF_ENDPOINT}"
echo "[CPU] HF_HOME     : ${HF_HOME}"
echo "[CPU] MODEL_REPO  : ${MODEL_REPO}"
echo "[CPU] MODEL_DIR   : ${MODEL_DIR}"
echo "============================================================"

# -------------------------------------------------------------------
# Step 1: conda prefix env
# -------------------------------------------------------------------
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
python -V
python -m pip install -U pip setuptools wheel -i "${PYPI}" --trusted-host pypi.org

# -------------------------------------------------------------------
# Step 2: build a pinned requirements list (GPU offline lockfile)
#   IMPORTANT: tokenizers and huggingface-hub MUST match Transformers 5.x.
# -------------------------------------------------------------------
REQ_FILE="${WHEELHOUSE}/requirements.offline.txt"
cat > "${REQ_FILE}" << 'EOF'
# Core: Transformers + exact compatible deps
transformers==5.2.0
huggingface-hub==1.4.1
tokenizers==0.22.2
safetensors>=0.4.3
numpy==1.26.4
packaging>=20.0
pyyaml>=5.1
regex!=2019.12.17
tqdm>=4.27
typer-slim

# VLM utils & common scientific stack used by your experiment
accelerate==1.3.0
sentencepiece==0.2.0
qwen-vl-utils==0.0.10
Pillow==11.1.0
pycocotools==2.0.8
pandas==2.2.3
scipy==1.14.1
scikit-learn==1.6.1
matplotlib==3.9.4
seaborn==0.13.2
av==14.1.0
EOF

echo "[CPU] requirements written: ${REQ_FILE}"
sed -n '1,120p' "${REQ_FILE}"

# -------------------------------------------------------------------
# Step 3: download wheels for ALL requirements (and their deps) from OFFICIAL PyPI
# -------------------------------------------------------------------
echo "[CPU] Downloading wheels into wheelhouse (PyPI)..."
python -m pip download -d "${WHEELHOUSE}" -r "${REQ_FILE}" \
  -i "${PYPI}" --trusted-host pypi.org

# Torch CUDA wheels from official PyTorch index
echo "[CPU] Downloading PyTorch CUDA wheels..."
TORCH_IDX="https://download.pytorch.org/whl/cu124"
python -m pip download -d "${WHEELHOUSE}" --index-url "${TORCH_IDX}" \
  torch==2.5.1 torchvision==0.20.1

# Sanity checks: ensure the critical wheels exist
echo "[CPU] Sanity check critical wheels..."
ls -lh "${WHEELHOUSE}"/transformers-5.2.0*.whl
ls -lh "${WHEELHOUSE}"/huggingface_hub-1.4.1*.whl
ls -lh "${WHEELHOUSE}"/tokenizers-0.22.2*.whl

# -------------------------------------------------------------------
# Step 4: download model snapshot using huggingface_hub Python API
# -------------------------------------------------------------------
echo "[CPU] Installing huggingface-hub==1.4.1 into CPU env for snapshot_download..."
python -m pip install -U "huggingface-hub==1.4.1" -i "${PYPI}" --trusted-host pypi.org

echo "[CPU] Downloading/resuming model snapshot..."
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

echo "[CPU] Model dir top files:"
ls -lh "${MODEL_DIR}" | head -n 30

echo "============================================================"
echo "[CPU] DONE."
echo "[CPU] Wheelhouse: ${WHEELHOUSE}"
echo "[CPU] Model dir : ${MODEL_DIR}"
echo "Next: run gpu.sh on GPU node (offline)."
echo "============================================================"
