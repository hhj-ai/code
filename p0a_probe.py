"""P0-a 架构探测：确认 grid/hook/映射/替换 全部可工作，否则直接报错。"""

import argparse, json, os, sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from model_loader import load, num_layers
from visual_token_map import bbox_to_token_indices, surrounding_indices, find_visual_range
from ced_core import HiddenCapture, TokenReplacer, replacing, js_div, ent, get_image_token_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--coco_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output", default="results/p0a_probe_report.json")
    args = ap.parse_args()

    rpt = {}

    # 加载模型
    print("=== 加载模型 ===")
    processor, model, cfg = load(args.model_dir, args.device, args.dtype)
    nl = num_layers(cfg, model)
    rpt["num_layers"] = nl
    print(f"  num_layers={nl}")

    # 找测试图
    img_dir = f"{args.coco_dir}/val2017"
    jpgs = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    assert jpgs, f"没有jpg: {img_dir}"
    image = Image.open(f"{img_dir}/{jpgs[0]}").convert("RGB")
    w, h = image.size
    print(f"  测试图: {jpgs[0]} ({w}×{h})")

    # ── Probe 1: Visual Encoder 结构 ──
    print("\n=== Probe 1: Grid 结构 ===")
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": "Describe."}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)

    assert "image_grid_thw" in inputs, "processor 输出里没有 image_grid_thw"
    g = inputs["image_grid_thw"]
    t, gh, gw = (g[0].tolist() if g.dim() > 1 else g.tolist())
    gh, gw = int(gh), int(gw)
    rpt["grid"] = [int(t), gh, gw]
    print(f"  grid: t={int(t)} h={gh} w={gw} total={int(t*gh*gw)}")

    img_tid = get_image_token_id(processor)
    vs, ve = find_visual_range(inputs["input_ids"], img_tid)
    n_vis = ve - vs
    rpt["n_visual_tokens"] = n_vis
    print(f"  image_token_id={img_tid}, visual tokens [{vs},{ve}) = {n_vis}")
    assert n_vis == int(t * gh * gw), f"grid({int(t*gh*gw)}) != sequence({n_vis})"
    print("  ✓ grid 匹配 sequence")

    # ── Probe 2: Hook ──
    print("\n=== Probe 2: Hook ===")
    test_layers = [0, nl//2, nl-1]
    inputs_gpu = {k: v.to(args.device) for k, v in inputs.items()}
    cap = HiddenCapture()
    cap.register(model, test_layers)
    with torch.no_grad(): model(**inputs_gpu)
    assert len(cap.data) == len(test_layers), f"hook 只捕获了 {len(cap.data)}/{len(test_layers)} 层"
    rpt["hidden_dim"] = cap.data[test_layers[0]].shape[-1]
    print(f"  ✓ 捕获 {len(cap.data)} 层, hidden_dim={rpt['hidden_dim']}")
    cap.remove()

    # ── Probe 3: BBox 映射 ──
    print("\n=== Probe 3: BBox 映射 ===")
    for name, bbox in [("center", [w*.25, h*.25, w*.5, h*.5]),
                        ("small", [w*.4, h*.4, w*.2, h*.2])]:
        idx = bbox_to_token_indices(bbox, w, h, gh, gw)
        sur = surrounding_indices(idx, gh, gw)
        assert idx, f"{name}: target tokens 为空"
        print(f"  {name}: target={len(idx)} surround={len(sur)}")
    print("  ✓ 映射正常")

    # ── Probe 4: Token 替换效果 ──
    print("\n=== Probe 4: 替换效果 ===")
    mid = n_vis // 2; q = max(1, n_vis // 8)
    tgt_rel = list(range(mid-q, mid+q))
    sur_rel = [i for i in range(n_vis) if i not in set(tgt_rel)][:len(tgt_rel)*4]
    tgt_abs = [vs + i for i in tgt_rel]
    sur_abs = [vs + i for i in sur_rel]

    with torch.no_grad():
        p1 = torch.softmax(model(**inputs_gpu).logits[:, -1, :].float(), -1)

    rep = TokenReplacer(); rep.register(model); rep.set(tgt_abs, sur_abs)
    with torch.no_grad(), replacing(rep):
        p2 = torch.softmax(model(**inputs_gpu).logits[:, -1, :].float(), -1)
    rep.remove()

    js = js_div(p1, p2).item()
    rpt["replacement_js"] = js
    assert js > 1e-6, f"替换后 JS={js}，太小，替换无效"
    print(f"  JS={js:.6f}  H_orig={ent(p1).item():.4f}  H_repl={ent(p2).item():.4f}")
    print("  ✓ 替换有效")

    # 保存
    rpt["status"] = "passed"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    json.dump(rpt, open(args.output, "w"), indent=2)
    print(f"\n{'='*50}")
    print(f"  P0-a ✓ 全部通过")
    print(f"  报告: {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
