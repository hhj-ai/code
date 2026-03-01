#!/bin/bash
# ============================================================================
# GPU Server Script for CED Phase 0
# Run on GPU server (NO internet, 8x H200, shared filesystem with CPU server)
# Creates conda env from local packages, then runs P0 experiments
# ============================================================================

set -e

BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare"
CODE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/code"
OUTPUT_DIR="${CODE_DIR}/results_p0"

MODEL_PATH="${BASE_DIR}/models/Qwen2.5-VL-7B-Instruct"
COCO_IMG_DIR="${BASE_DIR}/data/coco/images/val2017"
COCO_ANN_FILE="${BASE_DIR}/data/coco/annotations/instances_val2017.json"
PACKAGES_DIR="${BASE_DIR}/packages"

echo "============================================"
echo "CED Phase 0 - GPU Server Script"
echo "============================================"

# -------------------------------------------------------------------
# Pre-flight checks
# -------------------------------------------------------------------
echo "[0] Pre-flight checks..."
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}. Run cpu.sh first."
    exit 1
fi
if [ ! -f "${COCO_ANN_FILE}" ]; then
    echo "ERROR: COCO annotations not found. Run cpu.sh first."
    exit 1
fi
if [ ! -d "${PACKAGES_DIR}" ]; then
    echo "ERROR: Pip packages not found at ${PACKAGES_DIR}. Run cpu.sh first."
    exit 1
fi

# -------------------------------------------------------------------
# Step 1: Create conda environment
# -------------------------------------------------------------------
echo "[1/3] Setting up conda environment..."

# Initialize conda
eval "$(conda shell.bash hook)"

# Create or reuse env
if conda env list | grep -q "ced_p0"; then
    echo "Conda env ced_p0 exists, activating..."
else
    echo "Creating conda env ced_p0 with python 3.10.0..."
    conda create -n ced_p0 python=3.10.0 -y
fi
conda activate ced_p0

# -------------------------------------------------------------------
# Step 2: Install packages from local wheels (offline)
# -------------------------------------------------------------------
echo "[2/3] Installing packages from local cache..."

# Install torch first (CUDA 12.4)
pip install --no-index --find-links="${PACKAGES_DIR}" \
    torch==2.5.1 torchvision==0.20.1 2>/dev/null || \
pip install --no-index --find-links="${PACKAGES_DIR}" \
    torch torchvision

# Install transformers stack
pip install --no-index --find-links="${PACKAGES_DIR}" \
    transformers accelerate qwen-vl-utils huggingface-hub \
    safetensors tokenizers sentencepiece av

# Install scientific packages
pip install --no-index --find-links="${PACKAGES_DIR}" \
    numpy scipy scikit-learn matplotlib seaborn \
    Pillow pycocotools tqdm pandas

echo "Package installation complete."

# Verify key packages
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
import transformers
print(f'Transformers: {transformers.__version__}')
" || { echo "ERROR: Package verification failed"; exit 1; }

# -------------------------------------------------------------------
# Step 3: Run Phase 0 Experiments
# -------------------------------------------------------------------
echo "[3/3] Running CED Phase 0 experiments..."

mkdir -p "${OUTPUT_DIR}"

cd "${CODE_DIR}"

# Set environment variables
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0  # P0 only needs single GPU

echo ""
echo "=== P0-a: Architecture Probing ==="
python main.py \
    --mode probe \
    --model_path "${MODEL_PATH}" \
    --coco_img_dir "${COCO_IMG_DIR}" \
    --coco_ann_file "${COCO_ANN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/log_p0a.txt"

echo ""
echo "=== P0-b: CED Signal Validation ==="
python main.py \
    --mode validate \
    --model_path "${MODEL_PATH}" \
    --coco_img_dir "${COCO_IMG_DIR}" \
    --coco_ann_file "${COCO_ANN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_samples 500 \
    --layers logits 12 16 20 24 27 \
    --lambda_e_values 0.0 0.05 0.1 0.2 0.3 0.5 \
    2>&1 | tee "${OUTPUT_DIR}/log_p0b.txt"

echo ""
echo "============================================"
echo "Phase 0 experiments complete!"
echo "Results saved to: ${OUTPUT_DIR}/"
echo "============================================"
echo ""
echo "Key output files:"
ls -la "${OUTPUT_DIR}/"
echo ""
echo "Check log files for detailed results:"
echo "  P0-a: ${OUTPUT_DIR}/log_p0a.txt"
echo "  P0-b: ${OUTPUT_DIR}/log_p0b.txt"
echo "  Results JSON: ${OUTPUT_DIR}/ced_results.json"
echo "  Figures: ${OUTPUT_DIR}/*.png"
