"""p0b_validate.py — P0-b 完整 CED 验证实验。

核心实验（预计 2-3 天，单卡 A100/H200）：
1. 对 VQA 数据集逐样本计算 CED
2. 根据模型回答 + GT 分为四组：
   - correct_positive:  GT=yes, model=yes（真看了，答对了）
   - hallucination:     GT=no,  model=yes（没看到但幻觉了有）
   - correct_negative:  GT=no,  model=no （没有，也知道没有）
   - miss:              GT=yes, model=no （有但没看到）
3. 比较四组的 CED 分布差异
4. 跨层分析（logits + 中间层 16/20/24/28/32）
5. 跨任务一致性（existence / spatial / attribute / counting）
6. 公式消融（JS / CED / KL / cosine / 纯熵）
7. 通过标准：correct_positive vs hallucination 的 AUC > 0.85
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from model_loader import load_qwen3vl, get_num_layers
from visual_token_map import (
    bbox_to_merged_token_indices,
    get_surrounding_token_indices,
    find_visual_token_range_in_input_ids,
    visual_token_absolute_positions,
)
from ced_core import (
    CEDComputer,
    prepare_inputs,
    get_image_token_id,
)


# ─────────────────── 行为分组 ───────────────────

def classify_behavior(
    gt_answer: str,
    model_answer: str,
    gt_present: bool,
    task_type: str,
) -> str:
    """将样本分为四组。
    
    对 existence 类问题（yes/no）：
    - correct_positive:  GT有(yes) + 模型答有(yes)
    - hallucination:     GT无(no)  + 模型答有(yes)
    - correct_negative:  GT无(no)  + 模型答无(no)
    - miss:              GT有(yes) + 模型答无(no)
    
    对其他类型：
    - correct_positive:  答对且 gt_present=True
    - hallucination:     答错且 gt_present=True（但模型可能编造了不存在的信息）
    - correct_negative:  只用于 existence，其他类型不适用 → 归为 other
    - miss:              答错且 gt_present=True
    """
    gt_norm = gt_answer.strip().lower()
    pred_norm = model_answer.strip().lower()

    if task_type == "existence":
        gt_yes = gt_norm.startswith("yes") or gt_norm == "true"
        pred_yes = pred_norm.startswith("yes") or pred_norm == "true" or "yes" in pred_norm[:10]

        if gt_yes and pred_yes:
            return "correct_positive"
        elif not gt_yes and pred_yes:
            return "hallucination"
        elif not gt_yes and not pred_yes:
            return "correct_negative"
        else:  # gt_yes and not pred_yes
            return "miss"
    else:
        # 非 existence 类：简单判断答案是否匹配
        is_correct = (gt_norm in pred_norm) or (pred_norm in gt_norm)
        if gt_present and is_correct:
            return "correct_positive"
        elif gt_present and not is_correct:
            return "miss"  # 或 hallucination，取决于具体错误类型
        else:
            return "other"


# ─────────────────── 主实验 ───────────────────

def run_experiment(args):
    print("=" * 60)
    print("  P0-b: CED Validation Experiment")
    print("=" * 60)

    # ---- 加载模型 ----
    print("\n[1/4] Loading model ...")
    processor, model, cfg = load_qwen3vl(args.model_dir, args.device, args.dtype)
    num_layers = get_num_layers(cfg, model)
    print(f"  Model loaded. num_layers={num_layers}")

    # ---- 解析监控层 ----
    monitor_layers = []
    for l in args.layers.split(","):
        l = l.strip()
        if l == "logits":
            continue  # logits 层在 CEDComputer 中自动计算
        monitor_layers.append(int(l))
    # 确保不超出范围
    monitor_layers = [l for l in monitor_layers if l < num_layers]
    print(f"  Monitor layers: {monitor_layers}")

    # lambda_e 扫描值
    lambda_values = [float(x) for x in args.lambda_e_sweep.split(",")]
    print(f"  Lambda sweep: {lambda_values}")

    # ---- 初始化 CED 计算器 ----
    ced = CEDComputer(model, processor, monitor_layers, args.device)

    # ---- 获取 image token id ----
    img_token_id = get_image_token_id(processor)
    print(f"  Image token id: {img_token_id}")

    # ---- 加载 VQA 数据集 ----
    print("\n[2/4] Loading VQA dataset ...")
    samples = []
    with open(args.vqa_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"  Loaded {len(samples)} samples")

    # ---- 逐样本计算 ----
    print(f"\n[3/4] Running CED computation on {len(samples)} samples ...")
    os.makedirs(os.path.dirname(args.output_raw) or ".", exist_ok=True)
    out_f = open(args.output_raw, "w", encoding="utf-8")

    stats = {
        "total": 0, "processed": 0, "errors": 0,
        "behavior_counts": {},
        "task_type_counts": {},
    }
    t_start = time.time()

    for i, sample in enumerate(tqdm(samples, desc="CED")):
        stats["total"] += 1

        try:
            # 加载图像
            img_path = os.path.join(args.coco_image_dir, sample["image_file"])
            if not os.path.isfile(img_path):
                stats["errors"] += 1
                continue
            image = Image.open(img_path).convert("RGB")

            question = sample["question"]
            gt_answer = sample["answer"]
            task_type = sample["task_type"]
            bbox = sample["target_bbox"]
            gt_present = sample.get("gt_present", True)
            img_w = sample.get("image_width", image.size[0])
            img_h = sample.get("image_height", image.size[1])

            # 准备输入
            inputs = prepare_inputs(processor, image, question, args.device)

            # 获取 grid 信息
            if "image_grid_thw" not in inputs:
                stats["errors"] += 1
                continue
            grid_thw = inputs["image_grid_thw"]
            t, grid_h, grid_w = (
                grid_thw[0].tolist() if grid_thw.dim() > 1 else grid_thw.tolist()
            )
            grid_h, grid_w = int(grid_h), int(grid_w)

            # bbox → token 映射
            target_rel = bbox_to_merged_token_indices(
                bbox, img_w, img_h, grid_h, grid_w
            )
            surround_rel = get_surrounding_token_indices(
                target_rel, grid_h, grid_w, ring_width=2
            )

            if not target_rel or not surround_rel:
                stats["errors"] += 1
                continue

            # 转换为绝对位置
            target_abs = visual_token_absolute_positions(
                inputs["input_ids"], target_rel, img_token_id
            )
            surround_abs = visual_token_absolute_positions(
                inputs["input_ids"], surround_rel, img_token_id
            )

            if not target_abs or not surround_abs:
                stats["errors"] += 1
                continue

            # 生成模型回答
            model_answer = ced.generate_answer(inputs, max_new_tokens=32)

            # 行为分组
            behavior = classify_behavior(gt_answer, model_answer, gt_present, task_type)

            # 计算 CED
            ced_results = ced.compute_ced(
                inputs, target_abs, surround_abs, lambda_values
            )

            # 组装结果
            record = {
                "idx": i,
                "image_id": sample.get("image_id"),
                "image_file": sample["image_file"],
                "question": question,
                "gt_answer": gt_answer,
                "model_answer": model_answer,
                "task_type": task_type,
                "gt_present": gt_present,
                "behavior": behavior,
                "bbox": bbox,
                "n_target_tokens": len(target_abs),
                "n_surround_tokens": len(surround_abs),
                "grid_h": grid_h,
                "grid_w": grid_w,
                **ced_results,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["processed"] += 1
            stats["behavior_counts"][behavior] = stats["behavior_counts"].get(behavior, 0) + 1
            stats["task_type_counts"][task_type] = stats["task_type_counts"].get(task_type, 0) + 1

            # 定期打印进度
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t_start
                speed = stats["processed"] / elapsed
                print(f"  [{i+1}/{len(samples)}] processed={stats['processed']} "
                      f"errors={stats['errors']} speed={speed:.1f} samples/s")

        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 10:
                print(f"  [ERROR] Sample {i}: {e}")
            continue

    out_f.close()

    # ---- 汇总 ----
    print(f"\n[4/4] Generating summary ...")
    elapsed = time.time() - t_start
    stats["wall_time_s"] = elapsed
    stats["speed_samples_per_s"] = stats["processed"] / elapsed if elapsed > 0 else 0

    summary = {
        "config": {
            "model_dir": args.model_dir,
            "vqa_file": args.vqa_file,
            "device": args.device,
            "dtype": args.dtype,
            "monitor_layers": monitor_layers,
            "lambda_values": lambda_values,
        },
        "stats": stats,
    }

    os.makedirs(os.path.dirname(args.output_summary) or ".", exist_ok=True)
    with open(args.output_summary, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  P0-b Complete.")
    print(f"  Processed: {stats['processed']}/{stats['total']} (errors: {stats['errors']})")
    print(f"  Behavior distribution: {stats['behavior_counts']}")
    print(f"  Task distribution: {stats['task_type_counts']}")
    print(f"  Wall time: {elapsed:.0f}s ({stats['speed_samples_per_s']:.1f} samples/s)")
    print(f"  Raw output: {args.output_raw}")
    print(f"  Summary: {args.output_summary}")
    print(f"{'=' * 60}")

    # 清理
    ced.cleanup()


def main():
    parser = argparse.ArgumentParser(description="P0-b: CED Validation")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--vqa_file", type=str, required=True)
    parser.add_argument("--coco_image_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output_raw", type=str, default="results/p0b_raw.jsonl")
    parser.add_argument("--output_summary", type=str, default="results/p0b_summary.json")
    parser.add_argument("--layers", type=str, default="logits,16,20,24,28,32",
                        help="逗号分隔的监控层（logits 自动包含）")
    parser.add_argument("--lambda_e_sweep", type=str, default="0.0,0.05,0.1,0.2,0.3,0.5",
                        help="λ_e 扫描值")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
