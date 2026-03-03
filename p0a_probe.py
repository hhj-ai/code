"""p0a_probe.py — P0-a 架构探测。

目的（一天内完成，单卡）：
1. 确认 Qwen3-VL 的 visual token 结构（grid_h, grid_w, total_tokens）
2. 确认 2×2 merger 的降采样比例
3. 确认 hook 能正常截获 hidden states
4. 确认 token 替换后 output 确实发生变化
5. 用一张 COCO 图做端到端 CED 计算验证

如果这一步失败，后续全部不用做。
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent))

from model_loader import load_qwen3vl, get_num_layers
from visual_token_map import (
    get_visual_token_grid,
    bbox_to_merged_token_indices,
    get_surrounding_token_indices,
    find_visual_token_range_in_input_ids,
    visual_token_absolute_positions,
)
from ced_core import (
    CEDComputer,
    prepare_inputs,
    get_image_token_id,
    HiddenStateCapture,
)


def probe_visual_encoder(model, processor, image, device, report):
    """探测 visual encoder 的输出结构。"""
    print("\n=== Probe 1: Visual Encoder Structure ===")

    # 准备一个简单输入
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "Describe this image."},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)

    # 检查 image_grid_thw
    if "image_grid_thw" in inputs:
        grid_thw = inputs["image_grid_thw"]
        t, h, w = grid_thw[0].tolist() if grid_thw.dim() > 1 else grid_thw.tolist()
        report["image_grid_thw"] = [int(t), int(h), int(w)]
        report["total_merged_tokens"] = int(t * h * w)
        print(f"  image_grid_thw: t={int(t)}, h={int(h)}, w={int(w)}")
        print(f"  total merged visual tokens: {int(t * h * w)}")
    else:
        report["image_grid_thw"] = None
        print("  [WARN] No image_grid_thw in processor output")

    # 检查 pixel_values shape
    if "pixel_values" in inputs:
        pv_shape = list(inputs["pixel_values"].shape)
        report["pixel_values_shape"] = pv_shape
        print(f"  pixel_values shape: {pv_shape}")
    else:
        report["pixel_values_shape"] = None

    # 检查 input_ids 中的 image token
    input_ids = inputs["input_ids"]
    report["input_ids_length"] = input_ids.shape[-1]

    try:
        img_token_id = get_image_token_id(processor)
        report["image_token_id"] = img_token_id

        # 找 visual token 范围
        vis_start, vis_end = find_visual_token_range_in_input_ids(input_ids, img_token_id)
        n_visual = vis_end - vis_start
        report["visual_token_range"] = [vis_start, vis_end]
        report["n_visual_tokens_in_sequence"] = n_visual
        print(f"  image_token_id: {img_token_id}")
        print(f"  visual tokens in input_ids: [{vis_start}, {vis_end}) = {n_visual} tokens")

        # 验证：merged tokens 数 == input_ids 中的 visual token 数
        if "image_grid_thw" in inputs:
            expected = int(t * h * w)
            match = n_visual == expected
            report["grid_matches_sequence"] = match
            if match:
                print(f"  ✓ Grid ({expected}) matches sequence ({n_visual})")
            else:
                print(f"  ✗ Grid ({expected}) != sequence ({n_visual})")
    except Exception as e:
        report["image_token_id_error"] = str(e)
        print(f"  [ERROR] {e}")

    return inputs


def probe_hooks(model, inputs, device, report):
    """探测 hook 是否能正常工作。"""
    print("\n=== Probe 2: Hook Mechanism ===")

    inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
    num_layers = report.get("num_layers", 32)

    test_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    test_layers = [l for l in test_layers if l < num_layers]

    capture = HiddenStateCapture()
    try:
        capture.register(model, test_layers)
        with torch.no_grad():
            model(**inputs_gpu)

        captured_layers = sorted(capture.captured.keys())
        report["hook_captured_layers"] = captured_layers
        report["hook_works"] = len(captured_layers) == len(test_layers)

        for layer_idx in captured_layers:
            hs = capture.captured[layer_idx]
            print(f"  Layer {layer_idx}: shape={list(hs.shape)}, dtype={hs.dtype}")

        if captured_layers:
            report["hidden_dim"] = capture.captured[captured_layers[0]].shape[-1]
            print(f"  ✓ Hook mechanism works. Hidden dim = {report['hidden_dim']}")
        else:
            print("  ✗ No layers captured!")

    except Exception as e:
        report["hook_error"] = str(e)
        print(f"  [ERROR] Hook failed: {e}")
    finally:
        capture.remove_hooks()


def probe_token_replacement(model, processor, image, device, report):
    """探测 token 替换后 output 是否变化。"""
    print("\n=== Probe 3: Token Replacement Effect ===")

    # 准备输入
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": "What objects are in this image?"},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)
    inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

    try:
        img_token_id = get_image_token_id(processor)
        vis_start, vis_end = find_visual_token_range_in_input_ids(
            inputs["input_ids"], img_token_id
        )
        n_visual = vis_end - vis_start

        if n_visual < 4:
            report["replacement_error"] = f"Too few visual tokens: {n_visual}"
            print(f"  [ERROR] Too few visual tokens: {n_visual}")
            return

        # 目标：中心区域（模拟一个物体占据的位置）
        mid = n_visual // 2
        quarter = max(1, n_visual // 8)
        target_rel = list(range(mid - quarter, mid + quarter))
        surround_rel = [i for i in range(n_visual) if i not in set(target_rel)]
        surround_rel = surround_rel[:len(target_rel) * 4]  # 限制周围 token 数

        target_abs = [vis_start + i for i in target_rel]
        surround_abs = [vis_start + i for i in surround_rel]

        report["test_target_tokens"] = len(target_abs)
        report["test_surround_tokens"] = len(surround_abs)

        # 正常前向
        with torch.no_grad():
            out_orig = model(**inputs_gpu)
        logits_orig = out_orig.logits[:, -1, :].float()
        probs_orig = torch.softmax(logits_orig, dim=-1)

        # 替换后前向
        from ced_core import VisualTokenReplacer, replacement_context
        replacer = VisualTokenReplacer()
        replacer.register(model)
        replacer.set_replacement(target_abs, surround_abs)

        with torch.no_grad():
            with replacement_context(replacer):
                out_replaced = model(**inputs_gpu)
        logits_replaced = out_replaced.logits[:, -1, :].float()
        probs_replaced = torch.softmax(logits_replaced, dim=-1)

        replacer.remove_hook()

        # 计算差异
        from ced_core import js_divergence, entropy, cosine_distance

        js = js_divergence(probs_orig, probs_replaced).item()
        h_orig = entropy(probs_orig).item()
        h_replaced = entropy(probs_replaced).item()
        cos = cosine_distance(logits_orig, logits_replaced).item()

        report["replacement_js"] = js
        report["replacement_entropy_orig"] = h_orig
        report["replacement_entropy_replaced"] = h_replaced
        report["replacement_cosine_dist"] = cos
        report["replacement_works"] = js > 1e-6  # 替换后有变化

        print(f"  JS divergence:    {js:.6f}")
        print(f"  Entropy (orig):   {h_orig:.4f}")
        print(f"  Entropy (repl):   {h_replaced:.4f}")
        print(f"  Cosine distance:  {cos:.6f}")

        if js > 1e-6:
            print("  ✓ Token replacement produces measurable output change")
        else:
            print("  ✗ Token replacement has NO effect (problem!)")

    except Exception as e:
        report["replacement_error"] = str(e)
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()


def probe_bbox_mapping(image, processor, report):
    """探测 bbox → token index 映射。"""
    print("\n=== Probe 4: BBox → Token Index Mapping ===")

    grid_thw = report.get("image_grid_thw")
    if not grid_thw:
        print("  [SKIP] No grid info available")
        return

    t, grid_h, grid_w = grid_thw
    w, h = image.size  # PIL: (width, height)

    # 测试几个 bbox
    test_bboxes = [
        ("center", [w * 0.25, h * 0.25, w * 0.5, h * 0.5]),
        ("top-left", [0, 0, w * 0.3, h * 0.3]),
        ("bottom-right", [w * 0.7, h * 0.7, w * 0.3, h * 0.3]),
        ("small-center", [w * 0.4, h * 0.4, w * 0.2, h * 0.2]),
    ]

    mapping_results = []
    for name, bbox in test_bboxes:
        indices = bbox_to_merged_token_indices(bbox, w, h, grid_h, grid_w)
        surround = get_surrounding_token_indices(indices, grid_h, grid_w)
        ratio = len(indices) / (grid_h * grid_w) if grid_h * grid_w > 0 else 0

        result = {
            "name": name,
            "bbox": bbox,
            "n_target_tokens": len(indices),
            "n_surround_tokens": len(surround),
            "coverage_ratio": round(ratio, 3),
        }
        mapping_results.append(result)
        print(f"  {name}: {len(indices)} target tokens, {len(surround)} surround, "
              f"coverage={ratio:.1%}")

    report["bbox_mapping_tests"] = mapping_results
    report["bbox_mapping_works"] = all(r["n_target_tokens"] > 0 for r in mapping_results)


def main():
    parser = argparse.ArgumentParser(description="P0-a: Architecture Probe")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--coco_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--output", type=str, default="results/p0a_probe_report.json")
    args = parser.parse_args()

    report = {"status": "running", "probes": {}}

    # ---- 加载模型 ----
    print("=== Loading Model ===")
    processor, model, cfg = load_qwen3vl(args.model_dir, args.device, args.dtype)
    num_layers = get_num_layers(cfg, model)
    report["model_dir"] = args.model_dir
    report["num_layers"] = num_layers
    report["model_type"] = getattr(cfg, "model_type", "unknown")
    print(f"  Model loaded. num_layers={num_layers}, type={report['model_type']}")

    # ---- 找一张 COCO 图 ----
    coco_img_dir = os.path.join(args.coco_dir, "val2017")
    if not os.path.isdir(coco_img_dir):
        # 如果 coco_dir 直接就是图片目录
        coco_img_dir = args.coco_dir

    jpg_files = sorted([f for f in os.listdir(coco_img_dir) if f.endswith(".jpg")])
    if not jpg_files:
        print(f"[FATAL] No jpg files in {coco_img_dir}")
        report["status"] = "failed"
        report["error"] = "no_images"
    else:
        img_path = os.path.join(coco_img_dir, jpg_files[0])
        image = Image.open(img_path).convert("RGB")
        print(f"  Test image: {jpg_files[0]} ({image.size[0]}×{image.size[1]})")
        report["test_image"] = jpg_files[0]
        report["test_image_size"] = list(image.size)

        # ---- 运行各项探测 ----
        inputs = probe_visual_encoder(model, processor, image, args.device, report)
        probe_hooks(model, inputs, args.device, report)
        probe_bbox_mapping(image, processor, report)
        probe_token_replacement(model, processor, image, args.device, report)

        # ---- 汇总 ----
        all_ok = (
            report.get("hook_works", False)
            and report.get("replacement_works", False)
            and report.get("bbox_mapping_works", False)
            and report.get("grid_matches_sequence", False)
        )
        report["status"] = "passed" if all_ok else "partial"
        report["all_probes_passed"] = all_ok

    # ---- 输出 ----
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 50}")
    print(f"  P0-a Result: {'✓ ALL PASSED' if report.get('all_probes_passed') else '✗ SOME FAILED'}")
    print(f"  Report: {args.output}")
    print(f"{'=' * 50}")

    if not report.get("all_probes_passed"):
        # 打印失败的项
        for key in ["hook_works", "replacement_works", "bbox_mapping_works", "grid_matches_sequence"]:
            val = report.get(key)
            if val is not True:
                print(f"  ✗ {key} = {val}")


if __name__ == "__main__":
    main()
