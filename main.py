#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""main.py (v7, Qwen3-VL 正确加载版)

✅ 回顾上下文的所有坑并修好：
1) 你原工程实际在用 Qwen3-VL（报错出现 Qwen3VLConfig），之前默认误写成 Qwen2.5-VL 导致重新下载 16GB —— 已修：默认 repo=auto + fallback=Qwen/Qwen3-VL-8B-Instruct
2) `hf download --resume-download` 在你机器不支持 —— 已修：完全不依赖 hf CLI，统一用 snapshot_download
3) 以前 `list[str]` 在老 Python 报错 —— 已修：typing.List 兼容写法
4) Qwen3VLConfig 不能用 AutoModelForCausalLM —— 已修：检测到 qwen3_vl 配置就用 Qwen3VLForConditionalGeneration（或 MoE 版本）
5) 你日志里提示 `torch_dtype` deprecated —— 已修：优先使用 dtype=...，必要时回退 torch_dtype=...
6) CUDA 动态库抢先加载导致 torch import 炸 —— 由 cpu.sh/gpu.sh 负责设置 LD_LIBRARY_PATH（脚本里已做）

用法：
  - 默认（自动找本地 Qwen3-VL，找不到才下载 fallback）：
      python main.py --download-only
  - 强制使用某个本地目录（完全不下载）：
      python main.py --repo-id /abs/path/to/Qwen3-VL-dir
  - 指定 HF repo：
      python main.py --repo-id Qwen/Qwen3-VL-8B-Instruct --local-dir /path/to/save
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Any, Optional, Dict, List

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

    # 浅扫描：目录名包含 qwen3 和 vl
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
    # 1) 直接给了本地目录
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

    # 3) local_dir 已经完整就直接用
    if _has_weights(local_dir):
        return local_dir

    os.makedirs(local_dir, exist_ok=True)

    # 4) 下载（不依赖 hf CLI）
    kwargs: Dict[str, Any] = dict(
        repo_id=repo_id_or_path,
        repo_type="model",
        local_dir=local_dir,
    )
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    snapshot_download(**kwargs)

    if not _has_weights(local_dir):
        raise RuntimeError("Download finished but no weights found in: {}".format(local_dir))
    return local_dir


def _get_num_hidden_layers(cfg: Any, model: Optional[Any] = None) -> int:
    # Qwen3-VL 的 num_hidden_layers 在 text_config 里
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


def _import_transformers():
    """延迟导入，保持 download-only 阶段不触发 torch/transformers load。"""
    try:
        import torch
    except Exception as e:
        print("[FATAL] import torch 失败：{}".format(repr(e)), file=sys.stderr)
        raise

    from transformers import AutoProcessor, AutoConfig

    # Qwen3-VL 推荐的类（Transformers 官方文档）
    try:
        from transformers import Qwen3VLForConditionalGeneration  # type: ignore
    except Exception:
        Qwen3VLForConditionalGeneration = None

    # MoE 变体（如果你用的是 Qwen3-VL-30B-A3B 之类）
    try:
        from transformers import Qwen3VLMoeForConditionalGeneration  # type: ignore
    except Exception:
        Qwen3VLMoeForConditionalGeneration = None

    # 兜底：有些 VLM 会走 AutoModelForVision2Seq
    try:
        from transformers import AutoModelForVision2Seq  # type: ignore
    except Exception:
        AutoModelForVision2Seq = None

    from transformers import AutoModelForCausalLM

    return torch, AutoProcessor, AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoModelForVision2Seq, AutoModelForCausalLM


def _call_from_pretrained(cls, model_path: str, *, dtype, device_map, trust_remote_code: bool, **kwargs):
    """兼容 transformers 的 dtype/torch_dtype 过渡期。优先 dtype，不行再 torch_dtype。"""
    try:
        return cls.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
    except TypeError:
        # 老版本/某些类仍用 torch_dtype
        return cls.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs
        )


def load_model(model_path: str, device: str, dtype: str, trust_remote_code: bool):
    torch, AutoProcessor, AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoModelForVision2Seq, AutoModelForCausalLM = _import_transformers()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    if dtype == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = getattr(torch, dtype)

    device_map = "auto" if (device.startswith("cuda") and torch.cuda.is_available()) else None

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_type = getattr(cfg, "model_type", "").lower()

    # ✅ 关键修复：Qwen3-VL 不能用 AutoModelForCausalLM
    if "qwen3_vl" in model_type or cfg.__class__.__name__ in ("Qwen3VLConfig", "Qwen3VLMoeConfig"):
        if ("moe" in model_type) and (Qwen3VLMoeForConditionalGeneration is not None):
            model = _call_from_pretrained(
                Qwen3VLMoeForConditionalGeneration,
                model_path,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                ignore_mismatched_sizes=True,
            )
        elif Qwen3VLForConditionalGeneration is not None:
            model = _call_from_pretrained(
                Qwen3VLForConditionalGeneration,
                model_path,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                ignore_mismatched_sizes=True,
            )
        else:
            raise RuntimeError(
                "Detected Qwen3-VL config but transformers doesn't expose Qwen3VLForConditionalGeneration. "
                "Please upgrade transformers to the version that includes Qwen3-VL support."
            )
    else:
        # 非 Qwen3-VL：尽量走 V2S，否则走 CausalLM
        if AutoModelForVision2Seq is not None:
            model = _call_from_pretrained(
                AutoModelForVision2Seq,
                model_path,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                ignore_mismatched_sizes=True,
            )
        else:
            model = _call_from_pretrained(
                AutoModelForCausalLM,
                model_path,
                dtype=torch_dtype,
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
    p.add_argument("--repo-id", type=str, default="auto",
                   help="auto | HF repo id | local dir path")
    p.add_argument("--local-dir", type=str, default="../dataprepare/models/Qwen3-VL-8B-Instruct")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--preferred-root", action="append", default=[])
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

    preferred_roots: List[str] = []
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
        # 采用 Qwen3-VL 文档推荐的 chat-template 方式（只用 text 也可）
        import torch
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello World"}],
            }
        ]
        inputs = exp.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # 一些版本会带 token_type_ids，生成时可删掉
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(exp.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = exp.model.generate(**inputs, max_new_tokens=16)
        # 只取新增部分
        in_len = inputs["input_ids"].shape[-1]
        trimmed = generated_ids[:, in_len:]
        out_text = exp.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print("[SMOKE_TEST_OUTPUT]", out_text, flush=True)


if __name__ == "__main__":
    main()
