#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_dataset.py (v12)

修复点（针对你这次 n_total_written=0 的事故）：
1) 自动发现数据集时，显式排除：
   - conda_envs / site-packages / dist-info / sboms / .hf_home / huggingface
2) 候选文件会做“快速 schema 体检”：
   - 读取前 N 条样本，至少有一定比例能抽到 prompt（默认 >=20%）
   - 否则认为不是数据集（比如 SBOM json），跳过
3) 支持从命令行传多个 --data_root，只在这些目录里找（gpu.sh 默认只给 dataprepare/data 与 dataprepare/datasets）

输出：
- predictions.jsonl：逐条预测
- summary.json：总体统计
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
            path = resolve_path_maybe_relative(v, base_dir)
            if path.exists():
                return Image.open(path).convert("RGB")

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

_EXCLUDE_SUBSTRINGS = [
    "/conda_env", "/conda_envs/", "/site-packages/", "/dist-info/",
    "/sboms/", "/.hf_home/", "/huggingface/", "/__pycache__/",
]

def _is_excluded_path(p: Path) -> bool:
    sp = str(p).replace("\\", "/")
    return any(x in sp for x in _EXCLUDE_SUBSTRINGS)

def _quick_schema_healthcheck(p: Path, max_samples: int = 30, min_prompt_ratio: float = 0.2) -> Tuple[bool, float, int]:
    """
    读取前 max_samples 条（或更少），统计能抽到 prompt 的比例。
    - 低于 min_prompt_ratio => 基本不是你要的数据集（SBOM/metadata/json 配置之类）
    """
    n = 0
    n_prompt = 0
    try:
        it = read_jsonl(p) if p.suffix == ".jsonl" else read_json(p)
        for sample in it:
            if not isinstance(sample, dict):
                continue
            n += 1
            if pick_prompt(sample):
                n_prompt += 1
            if n >= max_samples:
                break
    except Exception:
        return (False, 0.0, 0)

    if n == 0:
        return (False, 0.0, 0)
    ratio = n_prompt / n
    return (ratio >= min_prompt_ratio, ratio, n)

def discover_dataset(roots: List[Path]) -> Path:
    patterns = ["*.jsonl", "*.json"]
    cand: List[Path] = []
    seen = set()

    for r in roots:
        if not r.exists():
            continue
        for pat in patterns:
            for p in r.rglob(pat):
                if not p.is_file():
                    continue
                if p.suffix not in [".jsonl", ".json"]:
                    continue
                if _is_excluded_path(p):
                    continue
                sp = str(p)
                if sp in seen:
                    continue
                seen.add(sp)
                cand.append(p)

    if not cand:
        raise FileNotFoundError(
            "No dataset file found under: "
            + ", ".join(str(x) for x in roots)
            + ". Provide --data /path/to/xxx.jsonl"
        )

    # 按 mtime 新到旧排序，但要先做 schema 体检，跳过垃圾候选
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for p in cand[:200]:  # 最多检查前 200 个候选，够了
        ok, ratio, n = _quick_schema_healthcheck(p)
        if ok:
            print(f"[OK] Auto-selected dataset: {p} (prompt_ratio={ratio:.2f} over {n} samples)")
            return p
        else:
            print(f"[SKIP] Not dataset-like: {p} (prompt_ratio={ratio:.2f} over {n} samples)")

    raise FileNotFoundError(
        "Found json/jsonl files, but none looked like a dataset with prompts. "
        "Please pass --data explicitly."
    )

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
def short_generate(proc, model, messages, max_new_tokens: int, temperature: float) -> str:
    inputs = proc.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
    in_len = inputs["input_ids"].shape[-1]
    trimmed = gen_ids[:, in_len:]
    return proc.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--data_root", type=str, action="append", default=[])
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out_pred", type=str, required=True)
    ap.add_argument("--out_summary", type=str, required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    m = _import_from_main()
    proc, model = m.load_model(args.model_path, device=args.device, dtype=args.dtype, trust_remote_code=True)

    if args.data:
        dataset_path = Path(args.data).expanduser().resolve()
        if not dataset_path.exists():
            raise FileNotFoundError(f"--data not found: {dataset_path}")
    else:
        roots = [Path(x).expanduser().resolve() for x in (args.data_root or [])]
        if not roots:
            # 兜底：仍然只搜相对安全的两个目录
            roots = [(Path.cwd() / "../dataprepare/data").resolve(),
                     (Path.cwd() / "../dataprepare/datasets").resolve()]
        dataset_path = discover_dataset(roots)

    base_dir = dataset_path.parent

    it = read_jsonl(dataset_path) if dataset_path.suffix == ".jsonl" else read_json(dataset_path)

    os.makedirs(Path(args.out_pred).parent, exist_ok=True)
    out_f = open(args.out_pred, "w", encoding="utf-8")

    n_total_written = 0
    n_scored_with_gt = 0
    n_em = 0
    n_with_cf = 0
    js_sum = 0.0
    lat_sum = 0.0

    t0 = time.time()

    for idx, sample in enumerate(it):
        if not isinstance(sample, dict):
            continue

        prompt = pick_prompt(sample)
        if not prompt:
            continue

        # 主图（可选）
        img = load_image_any(sample,
                             ["image", "image_path", "img", "img_path", "path", "image_file"],
                             base_dir)

        content = []
        if img is not None:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        gt = pick_gt(sample)

        t_s = time.time()
        pred = short_generate(proc, model, messages, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        latency = time.time() - t_s

        em = None
        if gt is not None:
            n_scored_with_gt += 1
            em = exact_match(pred, gt)
            if em:
                n_em += 1

        # 反事实图：如果 sample 自带
        cf_img = load_image_any(sample,
                                ["cf_image", "image_cf", "counterfactual_image", "cf_image_path"],
                                base_dir)
        js_val = None
        if cf_img is not None:
            try:
                messages_cf = [{"role": "user", "content": [{"type": "image", "image": cf_img},
                                                           {"type": "text", "text": prompt}]}]
                p1 = first_token_probs(proc, model, messages)
                p2 = first_token_probs(proc, model, messages_cf)
                js_val = float(js_divergence(p1, p2).item())
                n_with_cf += 1
                js_sum += js_val
            except Exception:
                js_val = None

        rec = {
            "idx": idx,
            "prompt": prompt,
            "pred": pred,
            "gt": gt,
            "exact_match": em,
            "latency_s": latency,
            "js_first_token": js_val,
            "has_image": img is not None,
            "has_cf_image": cf_img is not None,
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_total_written += 1
        lat_sum += latency

    out_f.close()

    wall = time.time() - t0
    summary = {
        "dataset_path": str(dataset_path),
        "model_path": str(Path(args.model_path).resolve()),
        "device": args.device,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "n_total_written": n_total_written,
        "n_scored_with_gt": n_scored_with_gt,
        "exact_match": (n_em / n_scored_with_gt) if n_scored_with_gt else None,
        "n_with_cf_image": n_with_cf,
        "js_first_token_mean": (js_sum / n_with_cf) if n_with_cf else None,
        "latency_mean_s": (lat_sum / n_total_written) if n_total_written else None,
        "wall_time_s": wall,
        "out_pred": str(Path(args.out_pred).resolve()),
    }

    os.makedirs(Path(args.out_summary).parent, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] wrote:", str(Path(args.out_pred).resolve()))
    print("[DONE] summary:", str(Path(args.out_summary).resolve()))
    print("[SUMMARY]", json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
