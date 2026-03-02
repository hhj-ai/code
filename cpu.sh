#!/bin/bash
# ============================================================
# CPU script (HAS INTERNET): build & download everything needed
# for running Qwen3-VL on OFFLINE GPU nodes.
#
# What it does:
#  1) Forces pip to use OFFICIAL PyPI (ignores any pip.conf)
#  2) Builds a transformers GIT wheel (no dependency resolution)
#  3) Downloads/updates required wheels into a shared wheelhouse
#  4) Downloads Qwen3-VL model snapshot via HF mirror endpoint
#
# You can run it repeatedly; it will resume downloads.
# ============================================================

set -euo pipefail

BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare"
ENV_DIR="${BASE_DIR}/conda_envs/ced_p0"
WHEELHOUSE="${BASE_DIR}/packages"
HF_HOME="${BASE_DIR}/.hf_cache"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_DIR="${BASE_DIR}/models/$(basename "${MODEL_REPO}")"

# ---- Force OFFICIAL PyPI (ignore company/internal pip config) ----
export PIP_CONFIG_FILE="/dev/null"
export PIP_INDEX_URL="https://pypi.org/simple"
export PIP_TRUSTED_HOST="pypi.org"
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- HF mirror (change to https://huggingface.co if desired) ----
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME_OVERRIDE:-$HF_HOME}"

mkdir -p "$WHEELHOUSE" "$HF_HOME" "$MODEL_DIR"

echo "============================================================"
echo "[CPU] BASE_DIR    : $BASE_DIR"
echo "[CPU] ENV_DIR     : $ENV_DIR"
echo "[CPU] WHEELHOUSE  : $WHEELHOUSE"
echo "[CPU] HF_ENDPOINT : $HF_ENDPOINT"
echo "[CPU] HF_HOME     : $HF_HOME"
echo "[CPU] MODEL_REPO  : $MODEL_REPO"
echo "[CPU] MODEL_DIR   : $MODEL_DIR"
echo "============================================================"

# ---- conda activate prefix env ----
__conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null || true)"
if [ -n "${__conda_setup}" ]; then
  eval "$__conda_setup"
else
  CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

if [ -d "$ENV_DIR" ] && [ -f "$ENV_DIR/bin/python" ]; then
  echo "[CPU] conda env exists: $ENV_DIR"
else
  echo "[CPU] creating conda env: $ENV_DIR"
  conda create --prefix "$ENV_DIR" --override-channels -c defaults python=3.10.0 -y
fi
conda activate "$ENV_DIR"
python -V
python -m pip install -U pip setuptools wheel -i https://pypi.org/simple --trusted-host pypi.org

# ---- (1) Build transformers git wheel WITHOUT deps ----
echo "[CPU] Building transformers wheel from git (no deps)..."
python -m pip wheel --no-deps -w "$WHEELHOUSE" \
  "git+https://github.com/huggingface/transformers.git"

# ---- (2) Download wheels needed for runtime/offline install ----
echo "[CPU] Downloading runtime wheels from OFFICIAL PyPI..."
python -m pip download -d "$WHEELHOUSE" -i https://pypi.org/simple --trusted-host pypi.org \
  "huggingface_hub[cli]==0.28.1" \
  "accelerate==1.3.0" \
  "safetensors==0.5.2" \
  "tokenizers==0.21.0" \
  "sentencepiece==0.2.0" \
  "qwen-vl-utils==0.0.10" \
  "numpy==1.26.4" "scipy==1.14.1" "scikit-learn==1.6.1" \
  "matplotlib==3.9.4" "seaborn==0.13.2" "Pillow==11.1.0" \
  "pycocotools==2.0.8" "tqdm==4.67.1" "pandas==2.2.3" "av==14.1.0" \
  "requests>=2.26.0" "packaging>=20.0" "pyyaml>=5.1" "regex!=2019.12.17" \
  "filelock" "fsspec" "typing-extensions" "jinja2" "sympy" "networkx"

# ---- PyTorch CUDA wheels (official PyTorch index) ----
echo "[CPU] Downloading PyTorch CUDA wheels..."
TORCH_IDX="https://download.pytorch.org/whl/cu124"
python -m pip download -d "$WHEELHOUSE" --index-url "$TORCH_IDX" \
  torch==2.5.1 torchvision==0.20.1

# ---- (3) Download model snapshot via Python API (supports local_dir_use_symlinks=False) ----
echo "[CPU] Downloading/resuming model snapshot: $MODEL_REPO"
python -m pip install -U "huggingface_hub[cli]==0.28.1" -i https://pypi.org/simple --trusted-host pypi.org

python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_REPO"]
local_dir = os.environ["MODEL_DIR"]
snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
print("[CPU] snapshot_download done:", repo_id, "->", local_dir)
PY

echo "[CPU] Sanity check model files:"
ls -lh "$MODEL_DIR" | head -n 20
echo "[CPU] Wheelhouse transformers wheels:"
ls -lh "$WHEELHOUSE"/transformers-*.whl | tail -n 5

echo "============================================================"
echo "[CPU] DONE. Now go to GPU node and run gpu_install_and_run.sh"
echo "============================================================"
