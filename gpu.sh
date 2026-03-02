#!/bin/bash
# ============================================================================
# GPU Server Script for CED Phase 0
# Run on GPU server (NO internet, 8x H200, shared filesystem with CPU server)
# Activates conda env from shared path, installs from local wheels, runs P0
# ============================================================================

set -e

BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare"
CODE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/code"
OUTPUT_DIR="${CODE_DIR}/results_p0"

ENV_DIR="${BASE_DIR}/conda_envs/ced_p0"
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

check_path() {
    if [ ! -e "$1" ]; then
        echo "ERROR: $2 not found at $1"
        echo "       Please run cpu.sh on the CPU server first."
        exit 1
    fi
    echo "  OK: $2"
}

check_path "${ENV_DIR}/bin/python"   "Conda environment"
check_path "${MODEL_PATH}/config.json" "Model weights"
check_path "${COCO_ANN_FILE}"         "COCO annotations"
check_path "${PACKAGES_DIR}"          "Pip packages"

# -------------------------------------------------------------------
# Step 1: Activate conda environment via prefix path
# -------------------------------------------------------------------
echo ""
echo "[1/3] Activating conda environment..."

# Initialize conda
__conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
    if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    else
        echo "ERROR: Cannot initialize conda."
        exit 1
    fi
fi

# Activate via full prefix path (no need for env name lookup)
conda activate "${ENV_DIR}"

# Verify we're in the right env (NOT system base)
ACTUAL_PYTHON=$(which python)
if [[ "${ACTUAL_PYTHON}" != "${ENV_DIR}"* ]]; then
    echo "WARNING: python resolves to ${ACTUAL_PYTHON}, expected ${ENV_DIR}/bin/python"
    echo "         Falling back to direct PATH manipulation..."
    export PATH="${ENV_DIR}/bin:$PATH"
    export CONDA_PREFIX="${ENV_DIR}"
fi
echo "  Python: $(which python) -> $(python --version)"

# -------------------------------------------------------------------
# Step 2: Install packages from local wheels (fully offline)
# -------------------------------------------------------------------
echo ""
echo "[2/3] Installing packages from local cache (offline)..."

# PyTorch (CUDA 12.4)
pip install --no-index --find-links="${PACKAGES_DIR}" \
    torch==2.5.1 torchvision==0.20.1 \
    2>/dev/null || \
pip install --no-index --find-links="${PACKAGES_DIR}" \
    torch torchvision

# Transformers ecosystem
pip install --no-index --find-links="${PACKAGES_DIR}" \
    transformers==4.48.3 \
    accelerate==1.3.0 \
    qwen-vl-utils==0.0.10 \
    huggingface-hub==0.27.1 \
    safetensors==0.5.2 \
    tokenizers==0.21.0 \
    sentencepiece==0.2.0 \
    av==14.1.0

# Scientific stack
pip install --no-index --find-links="${PACKAGES_DIR}" \
    numpy==1.26.4 \
    scipy==1.14.1 \
    scikit-learn==1.6.1 \
    matplotlib==3.9.4 \
    seaborn==0.13.2 \
    Pillow==11.1.0 \
    pycocotools==2.0.8 \
    tqdm==4.67.1 \
    pandas==2.2.3

echo ""
echo "  Package verification:"
python -c "
import torch
print(f'  PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}  GPUs={torch.cuda.device_count()}')
import transformers
print(f'  Transformers {transformers.__version__}')
import numpy, scipy, sklearn, matplotlib, PIL
print(f'  NumPy={numpy.__version__} SciPy={scipy.__version__} sklearn={sklearn.__version__}')
print(f'  Matplotlib={matplotlib.__version__} Pillow={PIL.__version__}')
" || { echo "ERROR: Package verification failed!"; exit 1; }

# -------------------------------------------------------------------
# Step 3: Run Phase 0 Experiments
# -------------------------------------------------------------------
echo ""
echo "[3/3] Running CED Phase 0 experiments..."

mkdir -p "${OUTPUT_DIR}"
cd "${CODE_DIR}"

# Offline mode for HuggingFace
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

# Single GPU is sufficient for P0
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "========== P0-a: Architecture Probing =========="
python main.py \
    --mode probe \
    --model_path "${MODEL_PATH}" \
    --coco_img_dir "${COCO_IMG_DIR}" \
    --coco_ann_file "${COCO_ANN_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/log_p0a.txt"

echo ""
echo "========== P0-b: CED Signal Validation =========="
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

# -------------------------------------------------------------------
# Done
# -------------------------------------------------------------------
echo ""
echo "============================================"
echo "Phase 0 complete!"
echo "============================================"
echo "Results:  ${OUTPUT_DIR}/"
echo ""
echo "Key files:"
ls -lh "${OUTPUT_DIR}/"*.json "${OUTPUT_DIR}/"*.png 2>/dev/null || echo "(check ${OUTPUT_DIR}/)"
echo ""
echo "Logs:"
echo "  P0-a: ${OUTPUT_DIR}/log_p0a.txt"
echo "  P0-b: ${OUTPUT_DIR}/log_p0b.txt"
echo "============================================"
