"""coco_vqa_gen.py — 从 COCO val2017 标注生成四类 VQA 问题。

四类问题（对应 P0-b 跨任务一致性验证）：
1. existence  — "Is there a <object> in the image?"（存在性）
2. spatial    — "Is the <A> to the left/right/above/below the <B>?"（空间关系）
3. attribute  — "What color/size is the <object>?"（属性识别）
4. counting   — "How many <object>s are in the image?"（计数）

每条样本包含：
- image_id, image_file: COCO 图片信息
- question, answer: VQA 问答
- task_type: 上述四类之一
- target_bbox: 目标物体的 [x, y, w, h]（用于 CED 计算时定位 visual token）
- gt_present: 目标物体是否真的在图中（用于行为分组）

在 CPU 服务器运行，不需要 GPU。
"""

from __future__ import annotations
import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

# pycocotools 在 cpu 环境已安装
from pycocotools.coco import COCO


# COCO 80 类名称
COCO_CATEGORIES = None  # 从 annotation 动态读取


def _relative_position(box_a, box_b) -> str:
    """判断 box_a 相对于 box_b 的空间关系。box = [x, y, w, h]"""
    cx_a = box_a[0] + box_a[2] / 2
    cy_a = box_a[1] + box_a[3] / 2
    cx_b = box_b[0] + box_b[2] / 2
    cy_b = box_b[1] + box_b[3] / 2

    dx = cx_a - cx_b
    dy = cy_a - cy_b

    if abs(dx) > abs(dy):
        return "to the right of" if dx > 0 else "to the left of"
    else:
        return "below" if dy > 0 else "above"


def generate_existence_questions(
    coco: COCO,
    img_anns: Dict[int, List[Any]],
    all_cat_names: List[str],
    max_per_image: int = 2,
) -> List[Dict]:
    """存在性问题：正例（图中有）+ 反例（图中无但文本先验会猜有）。"""
    samples = []
    for img_id, anns in img_anns.items():
        img_info = coco.imgs[img_id]
        present_cats = set()
        for ann in anns:
            cat_name = coco.cats[ann["category_id"]]["name"]
            present_cats.add(cat_name)

        # 正例
        for ann in anns[:max_per_image]:
            cat_name = coco.cats[ann["category_id"]]["name"]
            samples.append({
                "image_id": img_id,
                "image_file": img_info["file_name"],
                "question": f"Is there a {cat_name} in this image? Answer yes or no.",
                "answer": "yes",
                "task_type": "existence",
                "target_bbox": ann["bbox"],  # [x, y, w, h]
                "target_category": cat_name,
                "gt_present": True,
                "image_width": img_info["width"],
                "image_height": img_info["height"],
            })

        # 反例：选一个图中不存在的类
        absent = [c for c in all_cat_names if c not in present_cats]
        if absent:
            neg_cat = random.choice(absent)
            # 反例没有 target_bbox，用整图中心区域作为 control
            w, h = img_info["width"], img_info["height"]
            samples.append({
                "image_id": img_id,
                "image_file": img_info["file_name"],
                "question": f"Is there a {neg_cat} in this image? Answer yes or no.",
                "answer": "no",
                "task_type": "existence",
                "target_bbox": [w * 0.25, h * 0.25, w * 0.5, h * 0.5],  # 中心区域
                "target_category": neg_cat,
                "gt_present": False,
                "image_width": w,
                "image_height": h,
            })

    return samples


def generate_spatial_questions(
    coco: COCO,
    img_anns: Dict[int, List[Any]],
    max_per_image: int = 2,
) -> List[Dict]:
    """空间关系问题：两个物体间的相对位置。"""
    samples = []
    for img_id, anns in img_anns.items():
        if len(anns) < 2:
            continue
        img_info = coco.imgs[img_id]

        # 选面积较大的物体对（避免太小的标注）
        sorted_anns = sorted(anns, key=lambda a: a["area"], reverse=True)
        pairs_done = 0
        for i in range(len(sorted_anns)):
            if pairs_done >= max_per_image:
                break
            for j in range(i + 1, len(sorted_anns)):
                if pairs_done >= max_per_image:
                    break
                a, b = sorted_anns[i], sorted_anns[j]
                if a["area"] < 900 or b["area"] < 900:  # 太小跳过
                    continue
                cat_a = coco.cats[a["category_id"]]["name"]
                cat_b = coco.cats[b["category_id"]]["name"]
                rel = _relative_position(a["bbox"], b["bbox"])

                samples.append({
                    "image_id": img_id,
                    "image_file": img_info["file_name"],
                    "question": f"Is the {cat_a} {rel} the {cat_b}? Answer yes or no.",
                    "answer": "yes",
                    "task_type": "spatial",
                    "target_bbox": a["bbox"],
                    "target_category": cat_a,
                    "ref_bbox": b["bbox"],
                    "ref_category": cat_b,
                    "gt_present": True,
                    "image_width": img_info["width"],
                    "image_height": img_info["height"],
                })
                pairs_done += 1

    return samples


def generate_counting_questions(
    coco: COCO,
    img_anns: Dict[int, List[Any]],
) -> List[Dict]:
    """计数问题。"""
    samples = []
    for img_id, anns in img_anns.items():
        img_info = coco.imgs[img_id]

        # 按类别分组
        cat_counts: Dict[str, List[Any]] = defaultdict(list)
        for ann in anns:
            cat_name = coco.cats[ann["category_id"]]["name"]
            cat_counts[cat_name].append(ann)

        for cat_name, cat_anns in cat_counts.items():
            count = len(cat_anns)
            if count < 1 or count > 10:
                continue

            # target_bbox: 取所有该类物体的并集 bbox
            x_min = min(a["bbox"][0] for a in cat_anns)
            y_min = min(a["bbox"][1] for a in cat_anns)
            x_max = max(a["bbox"][0] + a["bbox"][2] for a in cat_anns)
            y_max = max(a["bbox"][1] + a["bbox"][3] for a in cat_anns)

            samples.append({
                "image_id": img_id,
                "image_file": img_info["file_name"],
                "question": f"How many {cat_name}s are in this image? Answer with a number.",
                "answer": str(count),
                "task_type": "counting",
                "target_bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "target_category": cat_name,
                "gt_present": True,
                "image_width": img_info["width"],
                "image_height": img_info["height"],
            })

    return samples


def generate_attribute_questions(
    coco: COCO,
    img_anns: Dict[int, List[Any]],
    max_per_image: int = 1,
) -> List[Dict]:
    """属性问题（简化版：大小判断，基于面积比）。
    注意：COCO 没有颜色标注，所以属性问题用 "相对大小" 替代。
    """
    samples = []
    for img_id, anns in img_anns.items():
        if len(anns) < 2:
            continue
        img_info = coco.imgs[img_id]
        sorted_anns = sorted(anns, key=lambda a: a["area"], reverse=True)

        if sorted_anns[0]["area"] < 2 * sorted_anns[-1]["area"]:
            continue  # 大小差异不够明显

        big_ann = sorted_anns[0]
        small_ann = sorted_anns[-1]
        big_cat = coco.cats[big_ann["category_id"]]["name"]
        small_cat = coco.cats[small_ann["category_id"]]["name"]

        if big_cat == small_cat:
            continue

        samples.append({
            "image_id": img_id,
            "image_file": img_info["file_name"],
            "question": f"Which is larger in this image, the {big_cat} or the {small_cat}?",
            "answer": big_cat,
            "task_type": "attribute",
            "target_bbox": big_ann["bbox"],
            "target_category": big_cat,
            "gt_present": True,
            "image_width": img_info["width"],
            "image_height": img_info["height"],
        })

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_images", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    ann_file = os.path.join(args.coco_dir, "annotations", "instances_val2017.json")
    if not os.path.isfile(ann_file):
        raise FileNotFoundError(f"COCO annotation not found: {ann_file}")

    print(f"[INFO] Loading COCO annotations from {ann_file} ...")
    coco = COCO(ann_file)

    all_cat_names = [cat["name"] for cat in coco.cats.values()]
    print(f"[INFO] {len(all_cat_names)} categories, {len(coco.imgs)} images")

    # 选取有标注的图片（按 image_id 排序取前 N 张）
    img_ids = sorted(coco.getImgIds())
    img_anns: Dict[int, List[Any]] = {}
    for img_id in img_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, iscrowd=False))
        if anns:
            img_anns[img_id] = anns
        if len(img_anns) >= args.max_images:
            break

    print(f"[INFO] Using {len(img_anns)} images with annotations")

    # 生成四类问题
    existence = generate_existence_questions(coco, img_anns, all_cat_names)
    spatial = generate_spatial_questions(coco, img_anns)
    counting = generate_counting_questions(coco, img_anns)
    attribute = generate_attribute_questions(coco, img_anns)

    all_samples = existence + spatial + counting + attribute
    random.shuffle(all_samples)

    print(f"[INFO] Generated {len(all_samples)} samples:")
    print(f"  existence: {len(existence)}")
    print(f"  spatial:   {len(spatial)}")
    print(f"  counting:  {len(counting)}")
    print(f"  attribute: {len(attribute)}")

    # 写出
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "ced_vqa_dataset.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 额外写一份统计
    stats = {
        "total": len(all_samples),
        "existence": len(existence),
        "spatial": len(spatial),
        "counting": len(counting),
        "attribute": len(attribute),
        "n_images": len(img_anns),
    }
    stats_path = os.path.join(args.output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[OK] Dataset written to {out_path}")
    print(f"[OK] Stats written to {stats_path}")


if __name__ == "__main__":
    main()
