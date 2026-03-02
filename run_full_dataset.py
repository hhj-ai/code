#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
run_full_dataset.py

默认跑“整个数据集”的评测脚本（离线可用，尽量兼容多种数据集 schema）。

它会：
- 自动定位数据集文件（如果你没传 --data）
- 流式读取样本（jsonl/json），逐条生成预测
- 如果样本里存在 GT（answer/label/output/target/...），做一个保守的 exact-match 统计
- 如果样本里存在反事实图（cf_image/image_cf/...），额外计算首 token 的 JS 散度与熵差
- 产出：
    predictions.jsonl：每条样本一行
    summary.json：整体统计（样本数、EM、平均 JS、平均耗时...）

注意：
- 这里的“反事实 JS”是一个可复现的弱版本（首 token 分布差异），用来证明你的“视觉反事实 -> 输出分布变化”信号存在且可批量跑。
- 你真正的 CED（中间层替换/遮挡）实验可以复用这个框架，把 per-sample 的 score 换成你的 CED score。
'''

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(f"Pillow not available in env: {e!r}")


# -----------------------------
# Metrics
# -----------------------------
def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


# -----------------------------
# Text normalization / scoring
# -----------------------------
_punct_re = re.compile(r"[\s\.\,\!\?\:\;\(\)\[\]\{\}\"\'\`]+", re.UNICODE)

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = _punct_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_choice_letter(s: str) -> Optional[str]:
    m = re.search(r"\b([ABCD])\b", s.upper())
    if m:
        return m.group(1)
    m = re.search(r"\b(option|answer)\s*[:\-]?\s*([ABCD])\b", s.lower())
    if m:
        return m.group(2).upper()
    return None

def extract_yesno(s: str) -> Optional[str]:
    s = normalize_text(s)
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    if s.startswith("true"):
        return "yes"
    if s.startswith("false"):
        return "no"
    return None

def exact_match(pred: str, gt: str) -> bool:
    gt_y = extract_yesno(gt)
    if gt_y is not None:
        return extract_yesno(pred) == gt_y

    gt_c = extract_choice_letter(gt)
    if gt_c is not None:
        return extract_choice_letter(pred) == gt_c

    return normalize_text(pred) == normalize_text(gt)


# -----------------------------
# Dataset I/O
# -----------------------------
def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def read_json(path: Path) -> Iterable[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
    elif isinstance(obj, dict):
        for key in ["data", "samples", "instances", "items"]:
            if key in obj and isinstance(obj[key], list):
                for x in obj[key]:
                    if isinstance(x, dict):
                        yield x
                return
        yield obj
    else:
        raise ValueError(f"Unsupported json top-level type: {type(obj)}")


def discover_dataset(default_roots: List[Path]) -> Path:
    cand: List[Path] = []
    patterns = [
        "*.jsonl", "*.json",
        "*dataset*.jsonl", "*dataset*.json",
        "*phase0*.jsonl", "*phase0*.json",
        "*ced*.jsonl", "*ced*.json",
        "*eval*.jsonl", "*eval*.json",
        "*test*.jsonl", "*test*.json",
    ]
    seen = set()
    for r in default_roots:
        if not r.exists():
            continue
        for pat in patterns:
            for p in r.rglob(pat):
                if not p.is_file():
                    continue
                if p.suffix not in [".jsonl", ".json"]:
                    continue
                sp = str(p)
                if "/.hf_home/" in sp or "/huggingface/" in sp:
                    continue
                if sp in seen:
                    continue
                seen.add(sp)
                cand.append(p)

    if not cand:
        raise FileNotFoundError(
            "No dataset file found under: "
            + ", ".join(str(x) for x in default_roots)
            + ". Provide --data /path/to/xxx.jsonl"
        )

    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]


def resolve_path_maybe_relative(p: str, base_dir: Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base_dir / pp).resolve()


def load_image_any(sample: Dict[str, Any], key_candidates: List[str], base_dir: Path) -> Optional[Image.Image]:
    for k in key_candidates:
        if k not in sample:
            continue
        v = sample[k]

        if isinstance(v, str) and v:
            # path
            path = resolve_path_maybe_relative(v, base_dir)
            if path.exists():
                return Image.open(path).convert("RGB")

            # try base64
            if v.startswith("data:image"):
                try:
                    b64 = v.split(",")[-1]
                    data = base64.b64decode(b64)
                    import io
                    return Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    pass

        if isinstance(v, dict):
            if "path" in v and isinstance(v["path"], str):
                path = resolve_path_maybe_relative(v["path"], base_dir)
                if path.exists():
                    return Image.open(path).convert("RGB")
            if "bytes" in v and isinstance(v["bytes"], str):
                try:
                    data = base64.b64decode(v["bytes"])
                    import io
                    return Image.open(io.BytesIO(data)).convert("RGB")
                except Exception:
                    pass

    return None


def pick_prompt(sample: Dict[str, Any]) -> Optional[str]:
    for k in ["prompt", "question", "instruction", "query", "text", "caption"]:
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    if isinstance(sample.get("input"), str) and isinstance(sample.get("instruction"), str):
        ins = sample["instruction"].strip()
        inp = sample["input"].strip()
        return ins + ("\n" + inp if inp else "")

    conv = sample.get("conversations") or sample.get("messages")
    if isinstance(conv, list) and conv:
        for msg in reversed(conv):
            if isinstance(msg, dict) and msg.get("role") in ("user", "human"):
                content = msg.get("content") or msg.get("value")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    return None


def pick_gt(sample: Dict[str, Any]) -> Optional[str]:
    for k in ["answer", "gt", "label", "output", "target", "response", "gold"]:
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, (int, float)):
            return str(v)

    v = sample.get("answers")
    if isinstance(v, list) and v:
        if isinstance(v[0], str):
            return v[0].strip()
    return None


# -----------------------------
# Model helpers
# -----------------------------
def _import_from_main():
    import importlib
    return importlib.import_module("main")


@torch.no_grad()
def first_token_probs(proc, model, messages) -> torch.Tensor:
    inputs = proc.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out = model(**inputs)
    logits = out.logits
    next_logits = logits[:, -1, :]
    probs = torch.softmax(next_logits.float(), dim=-1)
    return probs[0]


@torch.no_grad()
def generate_text(proc, model, messages, max_new_tokens: int, temperature: float = 0.0) -> str:
    inputs = proc.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 1e-6
    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=(temperature if do_sample else None),
    )
    in_len = inputs["input_ids"].shape[-1]
    trimmed = gen_ids[:, in_len:]
    return proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    p.add_argument("--data", type=str, default=None, help="dataset file (.jsonl/.json). If omitted, auto-discover.")
    p.add_argument("--max_samples", type=int, default=-1, help="<=0 means all")
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--out_pred", type=str, required=True)
    p.add_argument("--out_summary", type=str, required=True)
    p.add_argument("--save_every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_path).resolve()
    if not model_path.exists():
        print(f"[FATAL] model_path not found: {model_path}", file=sys.stderr)
        raise SystemExit(2)

    # Auto dataset discovery
    if args.data:
        data_path = Path(args.data).expanduser().resolve()
    else:
        roots = [
            Path.cwd() / "../dataprepare/data",
            Path.cwd() / "../dataprepare/datasets",
            Path.cwd() / "../dataprepare",
            Path.cwd() / "../data",
            Path.cwd() / "../datasets",
        ]
        data_path = discover_dataset([r.resolve() for r in roots])

    if not data_path.exists():
        print(f"[FATAL] dataset not found: {data_path}", file=sys.stderr)
        raise SystemExit(3)

    base_dir = data_path.parent

    # Load model
    m = _import_from_main()
    proc, model = m.load_model(str(model_path), device=args.device, dtype=args.dtype, trust_remote_code=True)

    out_pred = Path(args.out_pred).resolve()
    out_summary = Path(args.out_summary).resolve()
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    # Reader
    if data_path.suffix == ".jsonl":
        it = read_jsonl(data_path)
    elif data_path.suffix == ".json":
        it = read_json(data_path)
    else:
        raise ValueError(f"Unsupported dataset file: {data_path}")

    n = 0
    n_scored = 0
    n_em = 0
    js_vals: List[float] = []
    lat_vals: List[float] = []

    t_start = time.time()

    with out_pred.open("w", encoding="utf-8") as f_pred:
        for sample in it:
            if not isinstance(sample, dict):
                continue
            if args.max_samples > 0 and n >= args.max_samples:
                break

            sid = sample.get("id") or sample.get("uid") or sample.get("image_id") or sample.get("question_id") or n

            prompt = pick_prompt(sample)
            if not prompt:
                continue

            img = load_image_any(sample, ["image", "image_path", "img", "img_path", "image_file"], base_dir)
            if img is None:
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            else:
                messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                        {"type": "text", "text": prompt}]}]

            gt = pick_gt(sample)
            cf_img = load_image_any(sample, ["cf_image", "image_cf", "counterfactual_image", "edited_image"], base_dir)

            st = time.time()
            pred = generate_text(proc, model, messages, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
            latency = time.time() - st
            lat_vals.append(latency)

            em = None
            if gt is not None:
                try:
                    em = exact_match(pred, gt)
                    n_scored += 1
                    if em:
                        n_em += 1
                except Exception:
                    em = None

            js = None
            h0 = None
            h1 = None
            pred_cf = None
            if cf_img is not None:
                messages_cf = [{"role": "user", "content": [{"type": "image", "image": cf_img},
                                                           {"type": "text", "text": prompt}]}]
                try:
                    p0 = first_token_probs(proc, model, messages)
                    p1 = first_token_probs(proc, model, messages_cf)
                    js = float(js_divergence(p0, p1).item())
                    h0 = float(entropy(p0).item())
                    h1 = float(entropy(p1).item())
                    js_vals.append(js)
                except Exception:
                    js = None
                try:
                    pred_cf = generate_text(proc, model, messages_cf, max_new_tokens=min(args.max_new_tokens, 16), temperature=args.temperature)
                except Exception:
                    pred_cf = None

            rec: Dict[str, Any] = {
                "id": sid,
                "prompt": prompt,
                "pred": pred,
                "gt": gt,
                "exact_match": em,
                "latency_s": latency,
                "has_image": img is not None,
                "has_cf_image": cf_img is not None,
                "js_first_token": js,
                "entropy_first_token": h0,
                "entropy_first_token_cf": h1,
                "pred_cf": pred_cf,
            }
            f_pred.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

            if args.save_every > 0 and (n % args.save_every == 0):
                f_pred.flush()

            if n % 20 == 0:
                em_rate = (n_em / n_scored) if n_scored else None
                js_mean = (sum(js_vals) / len(js_vals)) if js_vals else None
                print(f"[PROG] n={n} scored={n_scored} em={em_rate} js_mean={js_mean}", flush=True)

    wall = time.time() - t_start
    em_rate = (n_em / n_scored) if n_scored else None
    js_mean = (sum(js_vals) / len(js_vals)) if js_vals else None
    lat_mean = (sum(lat_vals) / len(lat_vals)) if lat_vals else None

    summary: Dict[str, Any] = {
        "dataset_path": str(data_path),
        "model_path": str(model_path),
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "n_total_written": n,
        "n_scored_with_gt": n_scored,
        "exact_match": em_rate,
        "n_with_cf_image": len(js_vals),
        "js_first_token_mean": js_mean,
        "latency_mean_s": lat_mean,
        "wall_time_s": wall,
        "out_pred": str(out_pred),
    }

    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[DONE] wrote:", out_pred)
    print("[DONE] summary:", out_summary)
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
