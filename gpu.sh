#!/bin/bash
# =============================================================================
# gpu_resolve_deps.sh (GPU 无网)
# - 彻底解决 tokenizers==0.21.0 残留导致的依赖冲突
# - 离线安装 transformers==5.1.0 + huggingface-hub==1.4.1 + tokenizers==0.22.2
# - 验证：transformers import / is_offline_mode / AutoConfig(qwen3_vl)
# - 运行 main.py（P0-a / P0-b）
#
# 说明：
# pip 的 resolver 警告并不“修复”环境；你必须先卸载/强制重装正确版本。
# pip 卸载在少数情况下可能残留文件（例如某些非标准安装/元数据异常），所以这里加了
# 手动 rm site-packages 下 tokenizers/transformers/huggingface_hub 的保险清理。
# =============================================================================

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
echo "[GPU] ENV_DIR    : ${ENV_DIR}"
echo "[GPU] WHEELHOUSE : ${WHEELHOUSE}"
echo "[GPU] MODEL_DIR  : ${MODEL_DIR}"
echo "============================================================"

# conda activate prefix env
__conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null || true)"
if [ -n "${__conda_setup}" ]; then
  eval "$__conda_setup"
else
  CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
fi

conda deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true
conda activate "${ENV_DIR}"

echo "[GPU] CONDA_PREFIX=${CONDA_PREFIX}"
python - <<'PY'
import sys, os
print("sys.executable =", sys.executable)
print("CONDA_PREFIX  =", os.environ.get("CONDA_PREFIX"))
PY

# CUDA nvJitLink fix (避免系统 CUDA 抢库导致符号错误)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"

# offline flags
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

REQ_FILE="${WHEELHOUSE}/requirements.offline.txt"
if [ ! -f "${REQ_FILE}" ]; then
  echo "ERROR: ${REQ_FILE} not found. Run CPU script first."
  exit 1
fi

echo "[GPU] uninstall old/conflicting pkgs..."
python -m pip uninstall -y transformers huggingface-hub tokenizers || true

# last-resort cleanup (remove leftovers)
SITE_PKGS="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
echo "[GPU] site-packages: ${SITE_PKGS}"

rm -rf "${SITE_PKGS}/tokenizers" "${SITE_PKGS}/tokenizers-"*.dist-info 2>/dev/null || true
rm -rf "${SITE_PKGS}/huggingface_hub" "${SITE_PKGS}/huggingface_hub-"*.dist-info 2>/dev/null || true
rm -rf "${SITE_PKGS}/transformers" "${SITE_PKGS}/transformers-"*.dist-info 2>/dev/null || true

echo "[GPU] install torch wheels (offline)..."
python -m pip install --no-index --find-links="${WHEELHOUSE}" torch==2.5.1 torchvision==0.20.1 || true

echo "[GPU] install pinned compatible core stack (offline, force-reinstall)..."
python -m pip install --no-index --find-links="${WHEELHOUSE}" --force-reinstall   tokenizers==0.22.2   huggingface-hub==1.4.1   transformers==5.1.0

echo "[GPU] install remaining deps (offline) ..."
python -m pip install --no-index --find-links="${WHEELHOUSE}" -r "${REQ_FILE}"

echo "[GPU] verify versions/imports..."
python - <<'PY'
import tokenizers, transformers, huggingface_hub, torch
from huggingface_hub import is_offline_mode
print("tokenizers      :", tokenizers.__version__)
print("huggingface_hub :", huggingface_hub.__version__)
print("transformers    :", transformers.__version__)
print("torch           :", torch.__version__, "cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
print("HF offline mode :", is_offline_mode())
PY

echo "[GPU] verify qwen3_vl config recognized..."
python - <<'PY'
import os
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(os.environ["MODEL_DIR"], trust_remote_code=True)
print("config.model_type =", getattr(cfg, "model_type", None))
PY

# run experiments
mkdir -p "${OUT_DIR}"
cd "${CODE_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ATTN_IMPL="${ATTN_IMPL:-sdpa}"

echo ""
echo "========== P0-a: Architecture Probing =========="
python main.py   --mode probe   --model_path "${MODEL_DIR}"   --coco_img_dir "${COCO_IMG_DIR}"   --coco_ann_file "${COCO_ANN_FILE}"   --output_dir "${OUT_DIR}"   2>&1 | tee "${OUT_DIR}/log_p0a.txt"

echo ""
echo "========== P0-b: CED Signal Validation =========="
python main.py   --mode validate   --model_path "${MODEL_DIR}"   --coco_img_dir "${COCO_IMG_DIR}"   --coco_ann_file "${COCO_ANN_FILE}"   --output_dir "${OUT_DIR}"   --num_samples 500   --layers logits 12 16 20 24 27   --lambda_e_values 0.0 0.05 0.1 0.2 0.3 0.5   2>&1 | tee "${OUT_DIR}/log_p0b.txt"

echo "============================================================"
echo "[GPU] DONE. Results: ${OUT_DIR}"
echo "============================================================"
