#!/bin/bash
# ============================================================
# GPU script (NO INTERNET): install correct transformers git wheel
# from shared wheelhouse, then run main.py.
#
# Fixes:
#  - Ensures transformers >= Qwen3-VL support is installed (git wheel)
#  - Avoids accidentally using old transformers from wheelhouse
#  - Sets LD_LIBRARY_PATH to prefer wheel-bundled nvjitlink (CUDA lib mix fix)
# ============================================================

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
echo "[GPU] ENV_DIR    : $ENV_DIR"
echo "[GPU] WHEELHOUSE : $WHEELHOUSE"
echo "[GPU] MODEL_DIR  : $MODEL_DIR"
echo "============================================================"

# ---- conda activate prefix env ----
__conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null || true)"
if [ -n "${__conda_setup}" ]; then
  eval "$__conda_setup"
else
  CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

# Clean nested activations
conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate "$ENV_DIR"

echo "[GPU] CONDA_PREFIX=$CONDA_PREFIX"
echo "[GPU] python=$(which python)"
python -V

# ---- CUDA lib mix fix (prefer wheel-bundled nvjitlink) ----
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"

# ---- OFFLINE flags ----
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ---- Install from wheelhouse (offline) ----
echo "[GPU] Installing packages from local wheelhouse (offline)..."
pip install --no-index --find-links="$WHEELHOUSE" torch==2.5.1 torchvision==0.20.1 || true

# CRITICAL: install transformers git wheel explicitly
echo "[GPU] Installing transformers git wheel explicitly..."
pip uninstall -y transformers || true

# Pick the newest transformers wheel in wheelhouse
TF_WHL="$(ls -1t "$WHEELHOUSE"/transformers-*.whl | head -n 1)"
if [ -z "$TF_WHL" ]; then
  echo "ERROR: No transformers wheel found in $WHEELHOUSE"
  exit 1
fi
echo "[GPU] Using transformers wheel: $TF_WHL"
pip install --no-index --find-links="$WHEELHOUSE" "$TF_WHL"

# Install remaining deps (offline)
pip install --no-index --find-links="$WHEELHOUSE" \
  "huggingface_hub[cli]==0.28.1" accelerate==1.3.0 safetensors==0.5.2 tokenizers==0.21.0 sentencepiece==0.2.0 \
  qwen-vl-utils==0.0.10 numpy==1.26.4 scipy==1.14.1 scikit-learn==1.6.1 \
  matplotlib==3.9.4 seaborn==0.13.2 Pillow==11.1.0 pycocotools==2.0.8 tqdm==4.67.1 pandas==2.2.3 av==14.1.0 \
  requests packaging pyyaml regex filelock fsspec typing-extensions jinja2 sympy networkx

echo "[GPU] Verify versions:"
python - <<'PY'
import torch, transformers
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
PY

# ---- Run experiments ----
mkdir -p "$OUT_DIR"
cd "$CODE_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"

echo ""
echo "========== P0-a: Architecture Probing =========="
python main.py \
  --mode probe \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_file "$COCO_ANN_FILE" \
  --output_dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/log_p0a.txt"

echo ""
echo "========== P0-b: CED Signal Validation =========="
python main.py \
  --mode validate \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_file "$COCO_ANN_FILE" \
  --output_dir "$OUT_DIR" \
  --num_samples 500 \
  --layers logits 12 16 20 24 27 \
  --lambda_e_values 0.0 0.05 0.1 0.2 0.3 0.5 \
  2>&1 | tee "$OUT_DIR/log_p0b.txt"

echo "============================================================"
echo "[GPU] DONE. Results in: $OUT_DIR"
echo "============================================================"
