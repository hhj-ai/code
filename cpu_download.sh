#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# cpu_download.sh
# CPU服务器（有网）运行：下载模型、COCO数据、pip离线包、生成VQA数据集
#
# 路径与原 cpu.sh / gpu.sh 完全对齐：
#   CED_ENV  = .../hhj-train/dataprepare/conda_envs/ced_p0
#   模型     = $PWD/../dataprepare/models/Qwen3-VL-8B-Instruct
#   数据     = $PWD/../dataprepare/data/coco
#   VQA      = $PWD/../dataprepare/data/ced_vqa
#   pip缓存  = $PWD/../dataprepare/pip_wheels
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 路径配置（与原 cpu.sh / gpu.sh 对齐） ----------

CED_ENV="${CED_ENV:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/dataprepare/conda_envs/ced_p0}"
PY="$CED_ENV/bin/python"
PIP="$CED_ENV/bin/pip"

# 与原 cpu.sh 一致的 HF 缓存
export HF_HOME="${HF_HOME:-$PWD/.hf_home}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

# 模型（与原 cpu.sh 的 LOCAL_DIR 完全一致）
MODEL_DIR="${LOCAL_DIR:-$PWD/../dataprepare/models/Qwen3-VL-8B-Instruct}"
FALLBACK_REPO="${FALLBACK_REPO:-Qwen/Qwen3-VL-8B-Instruct}"
CACHE_DIR="${CACHE_DIR:-$HF_HUB_CACHE}"

# 数据目录（放在原 dataprepare/data 下）
COCO_DIR="$PWD/../dataprepare/data/coco"
VQA_DIR="$PWD/../dataprepare/data/ced_vqa"
PIP_CACHE="$PWD/../dataprepare/pip_wheels"

# ---------- LD_LIBRARY_PATH（与原 cpu.sh 完全一致） ----------

add_lib_dir () {
  local d="$1"
  if [ -d "$d" ]; then
    if [[ ":${LD_LIBRARY_PATH:-}:" != *":$d:"* ]]; then
      export LD_LIBRARY_PATH="$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
  fi
}
add_lib_dir "$CED_ENV/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/torch/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/cusparse/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/cublas/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
add_lib_dir "$CED_ENV/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"

echo "============================================"
echo "  CED P0 - CPU Download Script"
echo "============================================"
echo "  CED_ENV   : $CED_ENV"
echo "  MODEL_DIR : $MODEL_DIR"
echo "  COCO_DIR  : $COCO_DIR"
echo "  VQA_DIR   : $VQA_DIR"
echo "  PIP_CACHE : $PIP_CACHE"
echo "============================================"

# ---------- 1. 检查/创建 conda 环境 ----------
if [ ! -x "$PY" ]; then
    echo "[1/5] Creating conda environment at $CED_ENV ..."
    conda create -y -p "$CED_ENV" python=3.10
    echo "[OK] conda env created."
else
    echo "[1/5] Conda env exists: $CED_ENV"
fi
PY="$CED_ENV/bin/python"
PIP="$CED_ENV/bin/pip"

# ---------- 2. 安装 pip 包 + 下载离线 wheel ----------
echo "[2/5] Installing packages & downloading wheels ..."
mkdir -p "$PIP_CACHE"

"$PIP" install --upgrade pip
"$PIP" install -r "$SCRIPT_DIR/requirements.txt"

# 下载 wheel 到共享目录（GPU 离线安装用）
"$PIP" download \
    -r "$SCRIPT_DIR/requirements.txt" \
    -d "$PIP_CACHE" \
    --platform linux_x86_64 \
    --python-version 310 \
    --only-binary=:all: \
    || echo "[WARN] Some wheels may need source build, trying generic..."

"$PIP" download \
    -r "$SCRIPT_DIR/requirements.txt" \
    -d "$PIP_CACHE" \
    --no-binary=:none: \
    2>/dev/null || true

echo "[OK] Wheels cached in $PIP_CACHE"

# ---------- 3. 下载模型（与原 cpu.sh 逻辑一致） ----------
echo "[3/5] Downloading Qwen3-VL-8B-Instruct ..."
if [ -f "$MODEL_DIR/config.json" ] && find "$MODEL_DIR" -name "*.safetensors" | head -1 | grep -q .; then
    echo "[OK] Model already exists at $MODEL_DIR, skipping."
else
    mkdir -p "$MODEL_DIR"
    "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$FALLBACK_REPO',
    local_dir='$MODEL_DIR',
    cache_dir='$CACHE_DIR',
    repo_type='model',
)
print('[OK] Model downloaded.')
"
fi

# ---------- 4. 下载 COCO val2017 ----------
echo "[4/5] Downloading COCO val2017 ..."
mkdir -p "$COCO_DIR"

# 图片
if [ -d "$COCO_DIR/val2017" ] && [ "$(ls "$COCO_DIR/val2017"/*.jpg 2>/dev/null | head -1)" ]; then
    echo "[OK] COCO val2017 images exist, skipping."
else
    echo "  Downloading val2017 images (6.2GB) ..."
    cd "$COCO_DIR"
    wget -q --show-progress -c "http://images.cocodataset.org/zips/val2017.zip" -O val2017.zip
    unzip -q -o val2017.zip
    rm -f val2017.zip
    cd "$SCRIPT_DIR"
    echo "[OK] COCO val2017 images ready."
fi

# 标注
if [ -f "$COCO_DIR/annotations/instances_val2017.json" ]; then
    echo "[OK] COCO annotations exist, skipping."
else
    echo "  Downloading annotations ..."
    cd "$COCO_DIR"
    wget -q --show-progress -c "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -O annotations.zip
    unzip -q -o annotations.zip
    rm -f annotations.zip
    cd "$SCRIPT_DIR"
    echo "[OK] COCO annotations ready."
fi

# ---------- 5. 生成 VQA 数据集 ----------
echo "[5/5] Generating CED VQA dataset from COCO annotations ..."
mkdir -p "$VQA_DIR"

"$PY" -u "$SCRIPT_DIR/src/coco_vqa_gen.py" \
    --coco_dir "$COCO_DIR" \
    --output_dir "$VQA_DIR" \
    --max_images 2000 \
    --seed 42

echo ""
echo "============================================"
echo "  ALL DONE."
echo "    模型:   $MODEL_DIR"
echo "    COCO:   $COCO_DIR"
echo "    VQA:    $VQA_DIR"
echo "    Wheels: $PIP_CACHE"
echo ""
echo "  接下来到 GPU 服务器运行："
echo "    cd $(basename "$SCRIPT_DIR")"
echo "    bash gpu_run.sh"
echo "============================================"
