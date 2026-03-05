#!/usr/bin/env bash
set -euo pipefail
# ============================================================
# run.sh — CED P0 实验唯一入口（AutoDL 单机，大陆环境）
#
# 用法：
#   bash run.sh setup              # 装包+下模型+下COCO+生成VQA
#   bash run.sh probe              # P0-a 架构探测（几分钟）
#   bash run.sh validate           # P0-b CED验证（主实验）
#   bash run.sh analysis           # 分析已有结果
#   bash run.sh all                # probe → validate → analysis
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_DIR="$SCRIPT_DIR/models/Qwen3-VL-8B-Instruct"
COCO_DIR="$SCRIPT_DIR/data/coco"
VQA_DIR="$SCRIPT_DIR/data/ced_vqa"
VQA_FILE="$VQA_DIR/ced_vqa_dataset.jsonl"
RESULTS_DIR="$SCRIPT_DIR/results"

DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bfloat16}"

PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

MODE="${1:-all}"

# ============================================================
do_setup() {
    echo "===== 安装依赖 ====="
    pip install -q transformers accelerate huggingface_hub Pillow \
        numpy scipy scikit-learn matplotlib seaborn \
        pycocotools tqdm qwen-vl-utils -i "$PIP_MIRROR"

    python -c "
import torch, transformers
print(f'torch={torch.__version__} transformers={transformers.__version__}')
print(f'CUDA={torch.cuda.is_available()} GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"

    echo "===== 下载模型 ====="
    if [ -f "$MODEL_DIR/config.json" ]; then
        echo "模型已存在，跳过"
    else
        mkdir -p "$MODEL_DIR"
        python -c "
import os; os.environ['HF_ENDPOINT']='$HF_ENDPOINT'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='$MODEL_DIR', resume_download=True)
print('done')
"
    fi

    echo "===== 下载 COCO val2017 ====="
    mkdir -p "$COCO_DIR"
    if [ -d "$COCO_DIR/val2017" ] && ls "$COCO_DIR/val2017"/*.jpg &>/dev/null; then
        echo "图片已存在，跳过"
    else
        cd "$COCO_DIR"
        wget -q --show-progress -c "http://images.cocodataset.org/zips/val2017.zip" -O val2017.zip
        unzip -q -o val2017.zip && rm -f val2017.zip
        cd "$SCRIPT_DIR"
    fi
    if [ -f "$COCO_DIR/annotations/instances_val2017.json" ]; then
        echo "标注已存在，跳过"
    else
        cd "$COCO_DIR"
        wget -q --show-progress -c "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" -O ann.zip
        unzip -q -o ann.zip && rm -f ann.zip
        cd "$SCRIPT_DIR"
    fi

    echo "===== 生成 VQA 数据集 ====="
    if [ -f "$VQA_FILE" ]; then
        echo "VQA已存在，跳过"
    else
        mkdir -p "$VQA_DIR"
        python -u src/coco_vqa_gen.py --coco_dir "$COCO_DIR" --output_dir "$VQA_DIR"
    fi

    echo "===== setup 完成 ====="
}

# ============================================================
do_probe() {
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    mkdir -p "$RESULTS_DIR"
    python -u src/p0a_probe.py \
        --model_dir "$MODEL_DIR" \
        --coco_dir "$COCO_DIR" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --output "$RESULTS_DIR/p0a_probe_report.json"
}

do_validate() {
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    mkdir -p "$RESULTS_DIR"
    python -u src/p0b_validate.py \
        --model_dir "$MODEL_DIR" \
        --vqa_file "$VQA_FILE" \
        --coco_image_dir "$COCO_DIR/val2017" \
        --device "$DEVICE" \
        --dtype "$DTYPE" \
        --output_raw "$RESULTS_DIR/p0b_raw.jsonl" \
        --output_summary "$RESULTS_DIR/p0b_summary.json" \
        --layers "logits,16,20,24,28,32" \
        --lambda_e_sweep "0.0,0.05,0.1,0.2,0.3,0.5"
}

do_analysis() {
    mkdir -p "$RESULTS_DIR/figures"
    python -u src/analysis.py \
        --raw_file "$RESULTS_DIR/p0b_raw.jsonl" \
        --output_dir "$RESULTS_DIR"
}

# ============================================================
case "$MODE" in
    setup)    do_setup;;
    probe)    do_probe;;
    validate) do_validate;;
    analysis) do_analysis;;
    all)      do_probe; do_validate; do_analysis;;
    *)        echo "用法: bash run.sh {setup|probe|validate|analysis|all}"; exit 1;;
esac
