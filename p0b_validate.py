"""P0-b: 逐样本计算 CED，输出 JSONL 供 analysis.py 分析。"""

import argparse, json, os, sys, time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from model_loader import load, num_layers
from visual_token_map import bbox_to_token_indices, surrounding_indices, to_absolute
from ced_core import CEDComputer, prepare_inputs, get_image_token_id


def classify(gt, pred, gt_present, task_type):
    gn, pn = gt.strip().lower(), pred.strip().lower()
    if task_type == "existence":
        g_yes = gn.startswith("yes")
        p_yes = "yes" in pn[:20]
        if g_yes and p_yes:     return "correct_positive"
        if not g_yes and p_yes: return "hallucination"
        if not g_yes and not p_yes: return "correct_negative"
        return "miss"
    hit = gn in pn or pn in gn
    if gt_present and hit: return "correct_positive"
    if gt_present:         return "miss"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--vqa_file", required=True)
    ap.add_argument("--coco_image_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--output_raw", default="results/p0b_raw.jsonl")
    ap.add_argument("--output_summary", default="results/p0b_summary.json")
    ap.add_argument("--layers", default="logits,16,20,24,28,32")
    ap.add_argument("--lambda_e_sweep", default="0.0,0.05,0.1,0.2,0.3,0.5")
    args = ap.parse_args()

    print("=" * 60)
    print("  P0-b: CED 验证")
    print("=" * 60)

    processor, model, cfg = load(args.model_dir, args.device, args.dtype)
    nl = num_layers(cfg, model)
    layers = [int(l) for l in args.layers.split(",") if l.strip() != "logits" and int(l) < nl]
    lambdas = [float(x) for x in args.lambda_e_sweep.split(",")]
    img_tid = get_image_token_id(processor)
    ced = CEDComputer(model, processor, layers, args.device)

    samples = [json.loads(l) for l in open(args.vqa_file) if l.strip()]
    print(f"  {len(samples)} 条样本, 监控层={layers}, λ={lambdas}")

    os.makedirs(os.path.dirname(args.output_raw) or ".", exist_ok=True)
    out_f = open(args.output_raw, "w")
    stats = {"total": 0, "ok": 0, "err": 0, "behavior": {}, "task": {}}
    t0 = time.time()

    for i, s in enumerate(tqdm(samples, desc="CED")):
        stats["total"] += 1
        try:
            img_path = f"{args.coco_image_dir}/{s['image_file']}"
            assert os.path.isfile(img_path), f"图片不存在: {img_path}"
            image = Image.open(img_path).convert("RGB")

            inputs = prepare_inputs(processor, image, s["question"], args.device)
            assert "image_grid_thw" in inputs

            g = inputs["image_grid_thw"]
            _, gh, gw = (g[0].tolist() if g.dim() > 1 else g.tolist())
            gh, gw = int(gh), int(gw)
            iw, ih = s.get("image_width", image.size[0]), s.get("image_height", image.size[1])

            tgt = bbox_to_token_indices(s["target_bbox"], iw, ih, gh, gw)
            sur = surrounding_indices(tgt, gh, gw, ring=2)
            tgt_abs = to_absolute(inputs["input_ids"], tgt, img_tid)
            sur_abs = to_absolute(inputs["input_ids"], sur, img_tid)
            if not tgt_abs or not sur_abs: stats["err"] += 1; continue

            answer = ced.generate(inputs, max_new=32)
            beh = classify(s["answer"], answer, s.get("gt_present", True), s["task_type"])
            ced_r = ced.compute(inputs, tgt_abs, sur_abs, lambdas)

            rec = dict(idx=i, image_file=s["image_file"], question=s["question"],
                       gt_answer=s["answer"], model_answer=answer,
                       task_type=s["task_type"], gt_present=s.get("gt_present", True),
                       behavior=beh, n_tgt=len(tgt_abs), n_sur=len(sur_abs),
                       grid_h=gh, grid_w=gw, **ced_r)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["ok"] += 1
            stats["behavior"][beh] = stats["behavior"].get(beh, 0) + 1
            stats["task"][s["task_type"]] = stats["task"].get(s["task_type"], 0) + 1

        except Exception as e:
            stats["err"] += 1
            if stats["err"] <= 10: print(f"  [ERR] #{i}: {e}")

    out_f.close()
    elapsed = time.time() - t0

    summary = {"config": vars(args), "stats": {**stats, "time_s": elapsed}}
    os.makedirs(os.path.dirname(args.output_summary) or ".", exist_ok=True)
    json.dump(summary, open(args.output_summary, "w"), indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  ok={stats['ok']}/{stats['total']} err={stats['err']} time={elapsed:.0f}s")
    print(f"  行为: {stats['behavior']}")
    print(f"  任务: {stats['task']}")
    print(f"  输出: {args.output_raw}")
    print(f"{'='*60}")
    ced.cleanup()


if __name__ == "__main__":
    main()
