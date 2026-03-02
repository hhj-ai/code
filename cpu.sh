#!/bin/bash
# =============================================================================
# cpu_resolve_deps.sh (CPU 有网)
# - 彻底解决 Transformers 5.x 与 huggingface-hub / tokenizers 的版本冲突
# - 准备 GPU 无网离线安装所需的 wheelhouse
# - 下载 Qwen3-VL 模型到共享目录
#
# 关键点（官方约束）：
# - Transformers 5.x 依赖 huggingface-hub>=1.3.0 且 tokenizers>=0.22.0,<=0.23.0
#   （可在安装/版本检查中看到该约束；不满足会导入失败）
#
# 本脚本：
#  1) 强制 pip 只用官方 PyPI（忽略任何 pip.conf / 内网源）
#  2) 生成离线 lockfile: transformers==5.1.0 + huggingface-hub==1.4.1 + tokenizers==0.22.2
#  3) 下载 lockfile 中所有 wheels 到 wheelhouse
#  4) 下载模型 snapshot（可断点续传）
# =============================================================================

set -euo pipefail

BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare"
ENV_DIR="${BASE_DIR}/conda_envs/ced_p0"
WHEELHOUSE="${BASE_DIR}/packages"
HF_HOME_DIR="${BASE_DIR}/.hf_cache"

export MODEL_REPO="${MODEL_REPO:-Qwen/Qwen3-VL-8B-Instruct}"
export MODEL_DIR="${BASE_DIR}/models/$(basename "${MODEL_REPO}")"

# 强制官方 PyPI
export PIP_CONFIG_FILE="/dev/null"
export PIP_INDEX_URL="https://pypi.org/simple"
export PIP_TRUSTED_HOST="pypi.org"
export PIP_EXTRA_INDEX_URL=""
export PIP_NO_INPUT=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
PYPI="https://pypi.org/simple"

# HF endpoint（默认镜像；想用官方：HF_ENDPOINT=https://huggingface.co）
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

# conda activate prefix env
__conda_setup="$('conda' 'shell.bash' 'hook' 2>/dev/null || true)"
if [ -n "${__conda_setup}" ]; then
  eval "$__conda_setup"
else
  CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
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

# 生成 lockfile
REQ_FILE="${WHEELHOUSE}/requirements.offline.txt"
cat > "${REQ_FILE}" << 'EOF'
transformers==5.1.0
huggingface-hub==1.4.1
tokenizers==0.22.2
safetensors>=0.4.3
numpy==1.26.4
packaging>=20.0
pyyaml>=5.1
regex!=2019.12.17
tqdm>=4.27
typer-slim

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

echo "[CPU] lockfile: ${REQ_FILE}"

# 下载 wheels（官方 PyPI）
echo "[CPU] downloading wheels from OFFICIAL PyPI..."
python -m pip download -d "${WHEELHOUSE}" -r "${REQ_FILE}" -i "${PYPI}" --trusted-host pypi.org

# PyTorch CUDA wheels（官方 PyTorch index）
echo "[CPU] downloading torch wheels..."
TORCH_IDX="https://download.pytorch.org/whl/cu124"
python -m pip download -d "${WHEELHOUSE}" --index-url "${TORCH_IDX}" torch==2.5.1 torchvision==0.20.1

echo "[CPU] sanity check critical wheels:"
ls -lh "${WHEELHOUSE}"/transformers-5.1.0*.whl
ls -lh "${WHEELHOUSE}"/huggingface_hub-1.4.1*.whl
ls -lh "${WHEELHOUSE}"/tokenizers-0.22.2*.whl

# 下载模型 snapshot（需要 hub>=1.0 才有更完整工具链；我们已固定为 1.4.1）
python -m pip install -U "huggingface-hub==1.4.1" -i "${PYPI}" --trusted-host pypi.org

echo "[CPU] downloading/resuming model snapshot..."
python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo_id = os.environ["MODEL_REPO"]
local_dir = os.environ["MODEL_DIR"]
snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
print("snapshot_download done:", repo_id, "->", local_dir)
PY

echo "[CPU] model dir top files:"
ls -lh "${MODEL_DIR}" | head -n 30

echo "============================================================"
echo "[CPU] DONE. Run gpu_resolve_deps.sh on GPU node."
echo "============================================================"
