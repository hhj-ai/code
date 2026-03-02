#!/bin/bash
# ============================================================================
# CPU Server Download Script for CED Phase 0
# Run on CPU server (has internet access)
# ============================================================================
# Notes:
# - COCO downloads can be slow from some regions. This script:
#   * Prefers aria2c multi-connection download (respects datacenter proxy env)
#   * Forces IPv4 (often fixes slow IPv6 route issues)
#   * Supports resume/continue
# - This script DOES NOT copy/rename itself (to avoid git/versioning side-effects).
#   You decide where to place it in your repo.
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
# -------------------------------------------------------------------
echo "[1/5] Creating conda environment..."

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

conda activate "${ENV_DIR}"
echo "Activated: $(which python)"
echo "Python:   $(python --version)"

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
    "huggingface-hub==0.28.1" \
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
# Step 3: Download Model Weights
# -------------------------------------------------------------------
echo "[3/5] Downloading Qwen2.5-VL-7B-Instruct..."

MODEL_DIR="${BASE_DIR}/models/Qwen2.5-VL-7B-Instruct"
if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
    echo "Model already exists, skipping."
else
    # Ensure huggingface-cli available (pin to match offline wheelhouse)
    pip install "huggingface-hub==0.28.1" -i "${PYPI}" --trusted-host pypi.org
    huggingface-cli download \
        Qwen/Qwen2.5-VL-7B-Instruct \
        --local-dir "${MODEL_DIR}" \
        --local-dir-use-symlinks False
fi
echo "Model ready at ${MODEL_DIR}"

# -------------------------------------------------------------------
# Step 4: Download COCO val2017
#   - Prefer aria2c multi-connection
#   - Force IPv4 (avoid slow IPv6 routes)
#   - Keep using datacenter proxy env (http_proxy/https_proxy/all_proxy)
# -------------------------------------------------------------------
echo "[4/5] Downloading COCO val2017..."

COCO_DIR="${BASE_DIR}/data/coco"
COCO_ZIP_URL="http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_ZIP_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

COCO_CONN="${COCO_CONN:-16}"   # override: export COCO_CONN=8

ensure_aria2c() {
    if command -v aria2c >/dev/null 2>&1; then
        return 0
    fi
    echo "aria2c not found; trying to install (best-effort)..."
    if command -v apt-get >/dev/null 2>&1; then
        (sudo -n true >/dev/null 2>&1 && sudo apt-get update && sudo apt-get install -y aria2) || \
        (apt-get update && apt-get install -y aria2) || true
    fi
    if ! command -v aria2c >/dev/null 2>&1 && command -v conda >/dev/null 2>&1; then
        conda install -y -c conda-forge aria2 || true
    fi
    command -v aria2c >/dev/null 2>&1
}

download_file() {
    local url="$1"
    local out="$2"

    if ensure_aria2c; then
        # aria2c respects env proxies; also pass ALL_PROXY/all_proxy if present
        local proxy_arg=""
        if [ -n "${all_proxy}" ]; then
            proxy_arg="--all-proxy=${all_proxy}"
        elif [ -n "${ALL_PROXY}" ]; then
            proxy_arg="--all-proxy=${ALL_PROXY}"
        fi

        echo "Downloading with aria2c (connections=${COCO_CONN}, IPv4 only): ${url}"
        aria2c \
            --continue=true \
            --max-connection-per-server="${COCO_CONN}" \
            -x"${COCO_CONN}" -s"${COCO_CONN}" \
            -k1M \
            --disable-ipv6=true \
            --file-allocation=none \
            ${proxy_arg} \
            -o "${out}" \
            "${url}"
    else
        echo "Downloading with wget (IPv4 only): ${url}"
        wget -4 -c "${url}" -O "${out}"
    fi
}

# Images
if [ -d "${COCO_DIR}/images/val2017" ] && [ $(ls "${COCO_DIR}/images/val2017" 2>/dev/null | wc -l) -gt 4000 ]; then
    echo "COCO images already present, skipping."
else
    echo "Downloading val2017 images (~6GB)..."
    cd "${COCO_DIR}/images"
    download_file "${COCO_ZIP_URL}" "val2017.zip"
    unzip -o val2017.zip
    rm -f val2017.zip
fi

# Annotations
if [ -f "${COCO_DIR}/annotations/instances_val2017.json" ]; then
    echo "COCO annotations already present, skipping."
else
    echo "Downloading annotations..."
    cd "${COCO_DIR}"
    download_file "${COCO_ANN_ZIP_URL}" "annotations.zip"
    unzip -o annotations.zip
    rm -f annotations.zip
fi

# -------------------------------------------------------------------
# Step 5: Copy code to shared filesystem (ONLY main.py & gpu.sh)
#   - Keep cpu.sh out to avoid messing with your git workflow.
# -------------------------------------------------------------------
echo "[5/5] Copying code files (main.py, gpu.sh)..."

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
for f in main.py gpu.sh; do
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
