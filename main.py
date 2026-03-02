#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""main.py (v6)

你遇到的问题（结合上下文）：
- 你原工程在用 Qwen3-VL（日志出现 Qwen3VLConfig），但我之前脚本默认写成了 Qwen2.5-VL-7B，所以才会重新下 16GB。
- 你机器上的 `hf` CLI 版本不支持某些参数（如 --resume-download），所以不要依赖 CLI。

本版目标：
1) 默认用“自动本地复用”：优先在你项目常见目录里找已下载的 Qwen3-VL（有 config.json + *.safetensors/*.bin）。
   找到了就直接用本地目录，完全不下载。
2) 如果没找到，再从 Hub 下载 Qwen3-VL（默认 fallback：Qwen/Qwen3-VL-8B-Instruct）到一个稳定目录（默认 ../dataprepare/models/...）
3) 不再调用 hf CLI，统一使用 huggingface_hub.snapshot_download（避免 CLI 版本差异）。
4) 支持 --download-only：下载阶段不 import torch，避免 torch/CUDA so 问题把下载也卡死。
5) 兼容 Python 3.8+。

用法：
  - 默认（自动找本地 Qwen3-VL，否则下载 fallback）：
      python main.py --download-only
  - 直接指定本地目录（强制不下载）：
      python main.py --repo-id /abs/path/to/Qwen3-VL
  - 强制指定 Hub repo（会下载到 local-dir）：
      python main.py --repo-id Qwen/Qwen3-VL-8B-Instruct --local-dir /path/to/save

额外：
  - 若你下载速度慢/频繁限流，设置 HF_TOKEN 可提高限额/速度（在 shell 里 export HF_TOKEN=...）。
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Any, Optional, Dict, List, Tuple

from huggingface_hub import snapshot_download


def _has_weights(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    if not os.path.exists(os.path.join(d, "config.json")):
        return False
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.endswith(".safetensors") or fn.endswith(".bin"):
                return True
    return False


def _list_dir_children(parent: str) -> List[str]:
    try:
        return [os.path.join(parent, x) for x in os.listdir(parent)]
    except Exception:
        return []


def find_existing_qwen3vl(preferred_roots: List[str]) -> Optional[str]:
    """在常见目录中寻找已存在的 Qwen3-VL 模型目录。"""
    # 先尝试一些常见固定名字（命中率最高）
    common_names = [
        "Qwen3-VL-8B-Instruct",
        "Qwen3-VL-7B-Instruct",
        "Qwen3-VL-4B-Instruct",
        "Qwen3-VL-2B-Instruct",
    ]
    for r in preferred_roots:
        for name in common_names:
            p = os.path.join(r, name)
            if _has_weights(p):
                return p

    # 再做一次“浅扫描”：找名字里同时包含 qwen3 和 vl 的目录
    for r in preferred_roots:
        for child in _list_dir_children(r):
            bn = os.path.basename(child).lower()
            if ("qwen3" in bn) and ("vl" in bn) and os.path.isdir(child) and _has_weights(child):
                return child

    return None


def ensure_downloaded(repo_id_or_path: str,
                      local_dir: str,
                      cache_dir: Optional[str],
                      preferred_roots: List[str],
                      fallback_repo: str) -> str:
    """确保模型在本地可用，优先复用已有本地目录，避免重复下载。"""
    # 1) 如果直接给了本地目录
    if os.path.isdir(repo_id_or_path):
        if not _has_weights(repo_id_or_path):
            raise RuntimeError("Provided local dir has no weights/config.json: {}".format(repo_id_or_path))
        return repo_id_or_path

    # 2) auto：优先在 preferred_roots 中找
    if repo_id_or_path == "auto":
        hit = find_existing_qwen3vl(preferred_roots)
        if hit:
            print("[OK] Reusing existing local Qwen3-VL:", hit, flush=True)
            return hit
        repo_id_or_path = fallback_repo  # 没找到就用 fallback 下载

    # 3) local_dir 已经完整就直接用（同一个目录跑，不会重复下）
    if _has_weights(local_dir):
        return local_dir

    os.makedirs(local_dir, exist_ok=True)

    # 4) 下载（不依赖 hf CLI）
    kwargs: Dict[str, Any] = dict(
        repo_id=repo_id_or_path,
        repo_type="model",
        local_dir=local_dir,
    )
    # 注意：当 local_dir 指定时，hub 的文档说明 cache_dir 不会被用作主缓存，
    # 但传了也不影响；这里保留以兼容部分版本/内部实现。
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    snapshot_download(**kwargs)

    if not _has_weights(local_dir):
        raise RuntimeError("Download finished but no weights found in: {}".format(local_dir))
    return local_dir


def _get_num_hidden_layers(cfg: Any, model: Optional[Any] = None) -> int:
    if hasattr(cfg, "num_hidden_layers"):
        return int(getattr(cfg, "num_hidden_layers"))

    for sub_name in ["text_config", "llm_config", "language_config", "model_config", "transformer_config"]:
        sub = getattr(cfg, sub_name, None)
        if sub is not None and hasattr(sub, "num_hidden_layers"):
            return int(getattr(sub, "num_hidden_layers"))

    for alt in ["n_layer", "n_layers", "num_layers", "n_hidden_layers"]:
        if hasattr(cfg, alt):
            return int(getattr(cfg, alt))

    if model is not None:
        candidates = [
            (lambda m: len(m.model.layers)),
            (lambda m: len(m.model.model.layers)),
            (lambda m: len(m.transformer.h)),
            (lambda m: len(m.gpt_neox.layers)),
        ]
        for fn in candidates:
            try:
                return int(fn(model))
            except Exception:
                pass

    raise AttributeError("Cannot determine num_hidden_layers for this model/config.")


def _import_torch_and_transformers():
    try:
        import torch
    except Exception as e:
        print("[FATAL] import torch 失败：{}".format(repr(e)), file=sys.stderr)
        raise

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForVision2Seq
        has_v2s = True
    except Exception:
        AutoModelForVision2Seq = None
        has_v2s = False

    from transformers import AutoModelForCausalLM
    return torch, AutoProcessor, has_v2s, AutoModelForCausalLM, AutoModelForVision2Seq


def load_model(model_path: str, device: str, dtype: str, trust_remote_code: bool):
    torch, AutoProcessor, has_v2s, AutoModelForCausalLM, AutoModelForVision2Seq = _import_torch_and_transformers()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    if dtype == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    device_map = "auto" if (device.startswith("cuda") and torch.cuda.is_available()) else None

    if has_v2s and AutoModelForVision2Seq is not None:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            ignore_mismatched_sizes=True,
        )

    if device_map is None:
        model = model.to(device)

    model.eval()
    return processor, model


class CEDExperiment:
    def __init__(self, model_path: str, device: str, dtype: str, trust_remote_code: bool):
        self.processor, self.model = load_model(model_path, device=device, dtype=dtype, trust_remote_code=trust_remote_code)
        self.num_layers = _get_num_hidden_layers(self.model.config, self.model)


def parse_args():
    p = argparse.ArgumentParser()
    # ✅ 默认 auto：优先复用本地 Qwen3-VL
    p.add_argument("--repo-id", type=str, default="auto",
                   help="auto | HF repo id (e.g. Qwen/Qwen3-VL-8B-Instruct) | local dir path")
    # ✅ 默认落到 dataprepare/models（更符合你原工程的布局）
    p.add_argument("--local-dir", type=str, default="../dataprepare/models/Qwen3-VL-8B-Instruct")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--preferred-root", action="append", default=[],
                   help="Add a directory to search existing local models (can be repeated).")
    p.add_argument("--fallback-repo", type=str, default="Qwen/Qwen3-VL-8B-Instruct")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # 默认搜索根目录：dataprepare/models 和 code/models（结合你的工程结构）
    preferred_roots = []
    if args.preferred_root:
        preferred_roots.extend(args.preferred_root)
    preferred_roots.extend([
        "../dataprepare/models",
        "./models",
        "../models",
    ])

    model_path = ensure_downloaded(
        repo_id_or_path=args.repo_id,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        preferred_roots=preferred_roots,
        fallback_repo=args.fallback_repo,
    )
    print("[OK] Model ready at: {}".format(os.path.abspath(model_path)), flush=True)

    if args.download_only:
        return

    exp = CEDExperiment(model_path, device=args.device, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
    print("[OK] Loaded model. num_layers = {}".format(exp.num_layers), flush=True)

    if args.smoke_test:
        import torch
        with torch.no_grad():
            inputs = exp.processor(text="Hello World", return_tensors="pt")
            inputs = {k: v.to(exp.model.device) for k, v in inputs.items()}
            out = exp.model.generate(**inputs, max_new_tokens=16)
            txt = exp.processor.decode(out[0], skip_special_tokens=True)
            print("[SMOKE_TEST_OUTPUT]", txt, flush=True)


if __name__ == "__main__":
    main()
