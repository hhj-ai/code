#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""main.py (v7)

Fix: Qwen3-VL 不能用 AutoModelForCausalLM 加载。

你当前报错：
    ValueError: Unrecognized configuration class ... Qwen3VLConfig ... AutoModelForCausalLM

根因：
- Qwen3-VL 属于 image-text-to-text / VLM 类模型，应该用
  - Qwen3VLForConditionalGeneration（最直接）
  - 或 AutoModelForImageTextToText（Transformers v5 推荐的 AutoClass）
- 你原代码在 import 不到 AutoModelForVision2Seq 时，会回退到 AutoModelForCausalLM，
  于是触发上面的 ValueError。

本版改动：
- 优先尝试 Qwen3VLForConditionalGeneration / AutoModelForImageTextToText
- 兼容 Transformers v4.* 与 v5.*（dtype/torch_dtype 参数名差异、AutoModelForVision2Seq 在 v5 被移除）
- smoke-test 改为走 chat template（对 Qwen3-VL 更稳）

用法不变：cpu.sh / gpu.sh 继续可用。
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Any, Optional, Dict, List

from huggingface_hub import snapshot_download


# -------------------------
# Download helpers
# -------------------------

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


def ensure_downloaded(
    repo_id_or_path: str,
    local_dir: str,
    cache_dir: Optional[str],
    preferred_roots: List[str],
    fallback_repo: str,
) -> str:
    """确保模型在本地可用，优先复用已有本地目录，避免重复下载。"""
    # 1) 直接给了本地目录
    if os.path.isdir(repo_id_or_path):
        if not _has_weights(repo_id_or_path):
            raise RuntimeError(f"Provided local dir has no weights/config.json: {repo_id_or_path}")
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
        raise RuntimeError(f"Download finished but no weights found in: {local_dir}")
    return local_dir


# -------------------------
# Model loading helpers
# -------------------------

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
        print(f"[FATAL] import torch 失败：{repr(e)}", file=sys.stderr)
        raise

    import transformers
    from transformers import AutoProcessor, AutoConfig, AutoModel

    # v5 推荐：AutoModelForImageTextToText（VLM）
    try:
        from transformers import AutoModelForImageTextToText
    except Exception:
        AutoModelForImageTextToText = None

    # v4 常用：AutoModelForVision2Seq（v5 里已移除/不再导出）
    try:
        from transformers import AutoModelForVision2Seq
    except Exception:
        AutoModelForVision2Seq = None

    # 直连 Qwen3-VL 类（最稳）
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except Exception:
        Qwen3VLForConditionalGeneration = None
    try:
        from transformers import Qwen3VLMoeForConditionalGeneration
    except Exception:
        Qwen3VLMoeForConditionalGeneration = None

    # 仅用于非多模态/兜底（不要拿它加载 Qwen3-VL）
    try:
        from transformers import AutoModelForCausalLM
    except Exception:
        AutoModelForCausalLM = None

    return dict(
        torch=torch,
        transformers=transformers,
        AutoProcessor=AutoProcessor,
        AutoConfig=AutoConfig,
        AutoModel=AutoModel,
        AutoModelForImageTextToText=AutoModelForImageTextToText,
        AutoModelForVision2Seq=AutoModelForVision2Seq,
        Qwen3VLForConditionalGeneration=Qwen3VLForConditionalGeneration,
        Qwen3VLMoeForConditionalGeneration=Qwen3VLMoeForConditionalGeneration,
        AutoModelForCausalLM=AutoModelForCausalLM,
    )


def _safe_from_pretrained(model_cls, model_path: str, load_dtype, device_map, trust_remote_code: bool):
    """兼容 transformers v4/v5：dtype vs torch_dtype 参数名不同。"""
    common = dict(
        pretrained_model_name_or_path=model_path,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    # 先尝试 v5 风格：dtype=
    try:
        if load_dtype is None:
            return model_cls.from_pretrained(**common)
        return model_cls.from_pretrained(**common, dtype=load_dtype)
    except TypeError:
        # 回退 v4 风格：torch_dtype=
        if load_dtype is None:
            return model_cls.from_pretrained(**common)
        return model_cls.from_pretrained(**common, torch_dtype=load_dtype)


def _is_vlm_config(cfg) -> bool:
    mt = (getattr(cfg, "model_type", "") or "").lower()
    # 经验规则：qwen*_vl / *vl* 归到 VLM；更宽松一点不影响（我们会在尝试加载时再验证）
    return ("vl" in mt) or ("vision" in mt)


def load_model(model_path: str, device: str, dtype: str, trust_remote_code: bool):
    env = _import_torch_and_transformers()
    torch = env["torch"]
    AutoProcessor = env["AutoProcessor"]
    AutoConfig = env["AutoConfig"]

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_type = (getattr(cfg, "model_type", "") or "").lower()

    # dtype 选择：支持 auto / float16 / bfloat16 / float32
    if dtype == "auto":
        # 让 transformers 自己决定（与 Qwen 官方示例一致）
        load_dtype = "auto"
    else:
        load_dtype = getattr(torch, dtype)

    device_map = "auto" if (device.startswith("cuda") and torch.cuda.is_available()) else None

    # 构建候选加载器（从最合适到最兜底）
    candidates = []

    # Qwen3-VL 专用类（最优先）
    if "qwen3_vl" in model_type:
        if "moe" in model_type and env["Qwen3VLMoeForConditionalGeneration"] is not None:
            candidates.append(env["Qwen3VLMoeForConditionalGeneration"])
        if env["Qwen3VLForConditionalGeneration"] is not None:
            candidates.append(env["Qwen3VLForConditionalGeneration"])

    # 通用 VLM AutoClass（Transformers v5 推荐）
    if env["AutoModelForImageTextToText"] is not None:
        candidates.append(env["AutoModelForImageTextToText"])

    # v4 兼容
    if env["AutoModelForVision2Seq"] is not None:
        candidates.append(env["AutoModelForVision2Seq"])

    # 文本 LM（仅当不是 VLM 时才尝试）
    if (not _is_vlm_config(cfg)) and env["AutoModelForCausalLM"] is not None:
        candidates.append(env["AutoModelForCausalLM"])

    # 最终兜底：AutoModel
    candidates.append(env["AutoModel"])

    last_err: Optional[BaseException] = None
    model = None
    for cls in candidates:
        try:
            model = _safe_from_pretrained(cls, model_path, load_dtype, device_map, trust_remote_code)
            break
        except Exception as e:
            last_err = e
            continue

    if model is None:
        raise RuntimeError(
            "Failed to load model with all candidates. "
            f"model_type={model_type}, device={device}, dtype={dtype}. "
            f"Last error: {repr(last_err)}"
        )

    if device_map is None:
        model = model.to(device)

    model.eval()
    return processor, model


# -------------------------
# Experiment
# -------------------------

class CEDExperiment:
    def __init__(self, model_path: str, device: str, dtype: str, trust_remote_code: bool):
        self.processor, self.model = load_model(
            model_path, device=device, dtype=dtype, trust_remote_code=trust_remote_code
        )
        self.num_layers = _get_num_hidden_layers(self.model.config, self.model)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo-id",
        type=str,
        default="auto",
        help="auto | HF repo id (e.g. Qwen/Qwen3-VL-8B-Instruct) | local dir path",
    )
    p.add_argument("--local-dir", type=str, default="../dataprepare/models/Qwen3-VL-8B-Instruct")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument(
        "--preferred-root",
        action="append",
        default=[],
        help="Add a directory to search existing local models (can be repeated).",
    )
    p.add_argument("--fallback-repo", type=str, default="Qwen/Qwen3-VL-8B-Instruct")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="auto")  # auto | float16 | bfloat16 | float32
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def _smoke_test(processor, model):
    import torch

    # Qwen3-VL 官方推荐走 chat template；这里用纯文本消息，避免额外图片依赖。
    if hasattr(processor, "apply_chat_template"):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello World"}]}]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # 一些模型会带 token_type_ids；Qwen 系列常见可直接丢掉
        if isinstance(inputs, dict):
            inputs.pop("token_type_ids", None)
        try:
            inputs = inputs.to(model.device)  # BatchEncoding
        except Exception:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
    else:
        inputs = processor(text="Hello World", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=16)

    # transformers v5 的 decode 行为有变：尽量用 batch_decode
    if hasattr(processor, "batch_decode"):
        txt = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print("[SMOKE_TEST_OUTPUT]", txt[0] if isinstance(txt, list) else txt, flush=True)
    elif hasattr(processor, "decode"):
        print("[SMOKE_TEST_OUTPUT]", processor.decode(generated_ids[0], skip_special_tokens=True), flush=True)
    else:
        print("[SMOKE_TEST_OUTPUT] (no decode method)", flush=True)


def main():
    args = parse_args()

    preferred_roots = []
    if args.preferred_root:
        preferred_roots.extend(args.preferred_root)
    preferred_roots.extend(["../dataprepare/models", "./models", "../models"])

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
        _smoke_test(exp.processor, exp.model)


if __name__ == "__main__":
    main()
