#!/bin/bash
# ============================================================================
# CPU Server Download Script for CED Phase 0
# Run on CPU server (has internet access)
# Downloads: conda packages, pip wheels, model weights, COCO dataset
# ============================================================================

set -e

BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare"
CODE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/code"

echo "============================================"
echo "CED Phase 0 - CPU Download Script"
echo "Base dir: ${BASE_DIR}"
echo "Code dir: ${CODE_DIR}"
echo "============================================"

# -------------------------------------------------------------------
# Step 0: Create directory structure
# -------------------------------------------------------------------
mkdir -p "${BASE_DIR}/packages"
mkdir -p "${BASE_DIR}/models"
mkdir -p "${BASE_DIR}/data/coco/images"
mkdir -p "${BASE_DIR}/data/coco/annotations"
mkdir -p "${CODE_DIR}"

# -------------------------------------------------------------------
# Step 1: Create conda environment & download pip packages
# -------------------------------------------------------------------
echo "[1/4] Creating conda environment and downloading pip packages..."

# Create conda env (python 3.10.0)
conda create -n ced_p0 python=3.10.0 -y 2>/dev/null || echo "Env ced_p0 already exists, skipping creation."
eval "$(conda shell.bash hook)"
conda activate ced_p0

# Install pip first to ensure latest
pip install --upgrade pip

# Download all required packages as wheels to shared storage
# Core ML
pip download torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 \
    -d "${BASE_DIR}/packages/" || echo "WARN: torch download may have partial failure, check manually."

pip download \
    transformers==4.48.3 \
    accelerate==1.3.0 \
    qwen-vl-utils==0.0.10 \
    -d "${BASE_DIR}/packages/"

# Scientific & utilities
pip download \
    numpy==1.26.4 \
    scipy==1.14.1 \
    scikit-learn==1.6.1 \
    matplotlib==3.9.4 \
    seaborn==0.13.2 \
    Pillow==11.1.0 \
    pycocotools==2.0.8 \
    tqdm==4.67.1 \
    pandas==2.2.3 \
    av==14.1.0 \
    -d "${BASE_DIR}/packages/"

# HuggingFace tools
pip download \
    huggingface-hub==0.27.1 \
    safetensors==0.5.2 \
    tokenizers==0.21.0 \
    sentencepiece==0.2.0 \
    -d "${BASE_DIR}/packages/"

echo "Pip packages downloaded to ${BASE_DIR}/packages/"

# -------------------------------------------------------------------
# Step 2: Download Model Weights (Qwen2.5-VL-7B-Instruct)
# -------------------------------------------------------------------
echo "[2/4] Downloading Qwen2.5-VL-7B-Instruct model weights..."

MODEL_DIR="${BASE_DIR}/models/Qwen2.5-VL-7B-Instruct"
if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
    echo "Model already exists at ${MODEL_DIR}, skipping."
else
    pip install huggingface-hub
    huggingface-cli download \
        Qwen/Qwen2.5-VL-7B-Instruct \
        --local-dir "${MODEL_DIR}" \
        --local-dir-use-symlinks False
fi
echo "Model downloaded to ${MODEL_DIR}"

# -------------------------------------------------------------------
# Step 3: Download COCO val2017 dataset
# -------------------------------------------------------------------
echo "[3/4] Downloading COCO val2017 dataset..."

COCO_DIR="${BASE_DIR}/data/coco"

# Download val2017 images
if [ -d "${COCO_DIR}/images/val2017" ] && [ $(ls "${COCO_DIR}/images/val2017" | wc -l) -gt 4000 ]; then
    echo "COCO val2017 images already exist, skipping."
else
    echo "Downloading val2017 images (6.2GB)..."
    cd "${COCO_DIR}/images"
    wget -c http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
    unzip -o val2017.zip
    rm -f val2017.zip
fi

# Download annotations
if [ -f "${COCO_DIR}/annotations/instances_val2017.json" ]; then
    echo "COCO annotations already exist, skipping."
else
    echo "Downloading annotations..."
    cd "${COCO_DIR}/annotations"
    wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O annotations.zip
    unzip -o annotations.zip -d "${COCO_DIR}/"
    rm -f annotations.zip
fi

echo "COCO dataset ready at ${COCO_DIR}"

# -------------------------------------------------------------------
# Step 4: Clone / update code repo
# -------------------------------------------------------------------
echo "[4/4] Setting up code repository..."

if [ -d "${CODE_DIR}/.git" ]; then
    echo "Code repo already exists, pulling latest..."
    cd "${CODE_DIR}" && git pull
else
    git clone https://github.com/hhj-ai/code.git "${CODE_DIR}" 2>/dev/null || \
        echo "Git clone failed or repo not set up yet. Please copy code files manually to ${CODE_DIR}"
fi

# Copy main.py to code dir if not already there
# (In case user runs this before pushing to git)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "${SCRIPT_DIR}/main.py" ]; then
    cp "${SCRIPT_DIR}/main.py" "${CODE_DIR}/main.py"
    echo "Copied main.py to ${CODE_DIR}"
fi

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo "Download complete! Summary:"
echo "============================================"
echo "Packages:     ${BASE_DIR}/packages/ ($(ls ${BASE_DIR}/packages/ | wc -l) files)"
echo "Model:        ${BASE_DIR}/models/Qwen2.5-VL-7B-Instruct/"
echo "COCO images:  ${BASE_DIR}/data/coco/images/val2017/"
echo "COCO annots:  ${BASE_DIR}/data/coco/annotations/"
echo "Code:         ${CODE_DIR}/"
echo ""
echo "Next: Run gpu.sh on the GPU server."
echo "============================================"
