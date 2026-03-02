#!/bin/bash
# ============================================================================
# CPU Server Download Script for CED Phase 0
# Run on CPU server (has internet access)
# ============================================================================

set -e

BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare"
CODE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/code"
ENV_DIR="${BASE_DIR}/conda_envs/ced_p0"

echo "============================================"
echo "CED Phase 0 - CPU Download Script"
echo "Base dir: ${BASE_DIR}"
echo "Code dir: ${CODE_DIR}"
echo "Env dir:  ${ENV_DIR}"
echo "============================================"

# -------------------------------------------------------------------
# Step 0: Create directory structure
# -------------------------------------------------------------------
mkdir -p "${BASE_DIR}/packages"
mkdir -p "${BASE_DIR}/models"
mkdir -p "${BASE_DIR}/data/coco/images"
mkdir -p "${BASE_DIR}/data/coco/annotations"
mkdir -p "${BASE_DIR}/conda_envs"
mkdir -p "${CODE_DIR}"

# -------------------------------------------------------------------
# Step 1: Create conda environment
#   - Use --prefix so env lives on shared filesystem (both CPU & GPU can see it)
#   - Use --override-channels -c defaults to force official Anaconda channel
# -------------------------------------------------------------------
echo "[1/5] Creating conda environment..."

# Initialize conda for current shell
__conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
    if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    else
        echo "ERROR: Cannot initialize conda. Please check your conda installation."
        exit 1
    fi
fi

if [ -d "${ENV_DIR}" ] && [ -f "${ENV_DIR}/bin/python" ]; then
    echo "Conda env already exists at ${ENV_DIR}, skipping creation."
else
    echo "Creating conda env at ${ENV_DIR} (python=3.10.0, official channel only)..."
    conda create \
        --prefix "${ENV_DIR}" \
        --override-channels \
        -c defaults \
        python=3.10.0 \
        -y
fi

# Activate via prefix path (works regardless of envs_dirs config)
conda activate "${ENV_DIR}"
echo "Activated: $(which python)"
echo "Python:   $(python --version)"

# Upgrade pip from official source
pip install --upgrade pip \
    -i https://pypi.org/simple/ \
    --trusted-host pypi.org

# -------------------------------------------------------------------
# Step 2: Download pip packages (all from official PyPI / PyTorch)
# -------------------------------------------------------------------
echo "[2/5] Downloading pip packages..."

PYPI="https://pypi.org/simple/"
TORCH_IDX="https://download.pytorch.org/whl/cu124"
PKG="${BASE_DIR}/packages"

# --- PyTorch (CUDA 12.4) ---
pip download \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url "${TORCH_IDX}" \
    -d "${PKG}/"

# --- Transformers ecosystem ---
pip download \
    "transformers==4.48.3" \
    "accelerate==1.3.0" \
    "qwen-vl-utils==0.0.10" \
    "huggingface-hub==0.27.1" \
    "safetensors==0.5.2" \
    "tokenizers==0.21.0" \
    "sentencepiece==0.2.0" \
    -i "${PYPI}" --trusted-host pypi.org \
    -d "${PKG}/"

# --- Scientific & utilities ---
pip download \
    "numpy==1.26.4" \
    "scipy==1.14.1" \
    "scikit-learn==1.6.1" \
    "matplotlib==3.9.4" \
    "seaborn==0.13.2" \
    "Pillow==11.1.0" \
    "pycocotools==2.0.8" \
    "tqdm==4.67.1" \
    "pandas==2.2.3" \
    "av==14.1.0" \
    -i "${PYPI}" --trusted-host pypi.org \
    -d "${PKG}/"

echo "Downloaded $(ls ${PKG}/ | wc -l) package files to ${PKG}/"

# -------------------------------------------------------------------
# Step 3: Download Model Weights (via mirror, with completeness check)
# -------------------------------------------------------------------
echo "[3/5] Downloading Qwen2.5-VL-7B-Instruct (mirror + resume)..."

MODEL_DIR="${BASE_DIR}/models/Qwen2.5-VL-7B-Instruct"
mkdir -p "${MODEL_DIR}"

# Put HF cache on shared disk (avoids permission issues like writing to /models or other protected paths)
export HF_HOME="${BASE_DIR}/.hf_cache"
mkdir -p "${HF_HOME}"

# Use HF mirror endpoint (datacenter-friendly)
export HF_ENDPOINT="https://hf-mirror.com"

# Ensure CLI is available inside this env
# NOTE: keep hub pinned to the same version used offline on GPU side if you want (0.28.1).
pip install -U "huggingface_hub[cli]==0.28.1" -i "${PYPI}" --trusted-host pypi.org

# Check model completeness: require index + all 5 shards (Qwen2.5-VL-7B-Instruct uses 5 safetensors shards)
need_download=0
for f in \
  "config.json" \
  "model.safetensors.index.json" \
  "model-00001-of-00005.safetensors" \
  "model-00002-of-00005.safetensors" \
  "model-00003-of-00005.safetensors" \
  "model-00004-of-00005.safetensors" \
  "model-00005-of-00005.safetensors"; do
  if [ ! -f "${MODEL_DIR}/${f}" ]; then
    echo "[MODEL] missing: ${f}"
    need_download=1
  fi
done

if [ ${need_download} -eq 0 ]; then
  echo "[MODEL] already complete, skipping download."
else
  echo "[MODEL] downloading/resuming to ${MODEL_DIR} ..."
  if command -v hf >/dev/null 2>&1; then
    hf download Qwen/Qwen2.5-VL-7B-Instruct \
      --local-dir "${MODEL_DIR}" \
      --local-dir-use-symlinks False
  else
    # Backward-compatible CLI (still works even if deprecated)
    huggingface-cli download \
      Qwen/Qwen2.5-VL-7B-Instruct \
      --local-dir "${MODEL_DIR}" \
      --local-dir-use-symlinks False
  fi
fi

# Re-check
need_download=0
for f in \
  "model.safetensors.index.json" \
  "model-00001-of-00005.safetensors" \
  "model-00002-of-00005.safetensors" \
  "model-00003-of-00005.safetensors" \
  "model-00004-of-00005.safetensors" \
  "model-00005-of-00005.safetensors"; do
  if [ ! -f "${MODEL_DIR}/${f}" ]; then
    need_download=1
  fi
done
if [ ${need_download} -ne 0 ]; then
  echo "ERROR: Model download incomplete. Missing shards under: ${MODEL_DIR}"
  exit 1
fi

echo "Model ready at ${MODEL_DIR}"

# -------------------------------------------------------------------
# Step 4: Download COCO val2017
# -------------------------------------------------------------------
echo "[4/5] Downloading COCO val2017..."

COCO_DIR="${BASE_DIR}/data/coco"

# Images
if [ -d "${COCO_DIR}/images/val2017" ] && [ $(ls "${COCO_DIR}/images/val2017" 2>/dev/null | wc -l) -gt 4000 ]; then
    echo "COCO images already present, skipping."
else
    echo "Downloading val2017 images (~6GB)..."
    cd "${COCO_DIR}/images"
    wget -c http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
    unzip -o val2017.zip
    rm -f val2017.zip
fi

# Annotations
if [ -f "${COCO_DIR}/annotations/instances_val2017.json" ]; then
    echo "COCO annotations already present, skipping."
else
    echo "Downloading annotations..."
    cd "${COCO_DIR}"
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O annotations.zip
    unzip -o annotations.zip
    rm -f annotations.zip
fi

# -------------------------------------------------------------------
# Step 5: Copy code to shared filesystem
# -------------------------------------------------------------------
echo "[5/5] Copying code files..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for f in main.py gpu.sh cpu.sh; do
    if [ -f "${SCRIPT_DIR}/${f}" ]; then
        cp "${SCRIPT_DIR}/${f}" "${CODE_DIR}/${f}"
        echo "  Copied ${f} -> ${CODE_DIR}/"
    fi
done

# -------------------------------------------------------------------
# Done
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo "CPU download complete!"
echo "============================================"
echo "Conda env:  ${ENV_DIR}"
echo "Packages:   ${PKG}/ ($(ls ${PKG}/ | wc -l) files)"
echo "Model:      ${MODEL_DIR}/"
echo "COCO:       ${COCO_DIR}/"
echo "Code:       ${CODE_DIR}/"
echo ""
echo ">>> Next step: ssh to GPU server, then run:"
echo ">>>   cd ${CODE_DIR} && bash gpu.sh"
echo "============================================"
