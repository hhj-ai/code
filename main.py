#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""main.py (v8)

v7 已经修复了：Qwen3-VL 正确的模型类（Qwen3VLForConditionalGeneration）+ dtype 参数 + 不用 hf CLI + auto 复用本地。
你现在遇到的 `Killed`：
  - 不是 Python traceback，而是 OS/Scheduler 给进程发了 SIGKILL（通常是 OOM killer / cgroup memory limit）。
  - 你在 CPU 上用 float32 加载 8B 级模型，权重本身就 ~32GB（还没算额外开销与峰值），很容易被杀。

v8 目标：让 CPU 脚本“默认不做危险动作”，并提供可选的 CPU 低内存加载路径（极慢但能活）。
- 默认建议：CPU 只做 --download-only（避免被 kill）
- 若确实要 CPU 载入：
    * 默认 dtype 改为 bfloat16（内存减半）
    * 支持 --device-map auto + --max-cpu-mem + --offload-folder，将一部分权重 offload 到磁盘（更省内存，但更慢）
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
    if os.path.isdir(repo_id_or_path):
        if not _has_weights(repo_id_or_path):
            raise RuntimeError("Provided local dir has no weights/config.json: {}".format(repo_id_or_path))
        return repo_id_or_path

    if repo_id_or_path == "auto":
        hit = find_existing_qwen3vl(preferred_roots)
        if hit:
            print("[OK] Reusing existing local Qwen3-VL:", hit, flush=True)
            return hit
        repo_id_or_path = fallback_repo

    if _has_weights(local_dir):
        return local_dir

    os.makedirs(local_dir, exist_ok=True)
    kwargs: Dict[str, Any] = dict(repo_id=repo_id_or_path, repo_type="model", local_dir=local_dir)
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
        for fn in [
            (lambda m: len(m.model.layers)),
            (lambda m: len(m.model.model.layers)),
            (lambda m: len(m.transformer.h)),
            (lambda m: len(m.gpt_neox.layers)),
        ]:
            try:
                return int(fn(model))
            except Exception:
                pass
    raise AttributeError("Cannot determine num_hidden_layers for this model/config.")


def _import_transformers():
    try:
        import torch
    except Exception as e:
        print("[FATAL] import torch 失败：{}".format(repr(e)), file=sys.stderr)
        raise

    from transformers import AutoProcessor, AutoConfig

    try:
        from transformers import Qwen3VLForConditionalGeneration  # type: ignore
    except Exception:
        Qwen3VLForConditionalGeneration = None

    try:
        from transformers import Qwen3VLMoeForConditionalGeneration  # type: ignore
    except Exception:
        Qwen3VLMoeForConditionalGeneration = None

    try:
        from transformers import AutoModelForVision2Seq  # type: ignore
    except Exception:
        AutoModelForVision2Seq = None

    from transformers import AutoModelForCausalLM

    return torch, AutoProcessor, AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoModelForVision2Seq, AutoModelForCausalLM


def _call_from_pretrained(cls, model_path: str, *, dtype, device_map, trust_remote_code: bool, **kwargs):
    # 优先 dtype（新接口），不行再 torch_dtype（老接口）
    try:
        return cls.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
    except TypeError:
        return cls.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            **kwargs
        )


def _parse_max_memory(s: Optional[str]) -> Optional[Dict[str, str]]:
    # 形如: "24GiB" or "24000MiB"
    if not s:
        return None
    s = s.strip()
    return {"cpu": s, "disk": "200GiB"}  # disk 给大一点，避免再 OOM


def load_model(model_path: str,
               device: str,
               dtype: str,
               trust_remote_code: bool,
               device_map_arg: Optional[str],
               max_cpu_mem: Optional[str],
               offload_folder: Optional[str]):
    torch, AutoProcessor, AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoModelForVision2Seq, AutoModelForCausalLM = _import_transformers()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    if dtype == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            # CPU 默认 bfloat16 更省内存；不支持也能存权重，只是计算可能慢/有些 op 回退
            torch_dtype = torch.bfloat16
    else:
        torch_dtype = getattr(torch, dtype)

    # device_map 逻辑：
    # - GPU 默认 "auto"
    # - CPU 默认 None（意味着全放 CPU 内存） -> 但可通过 --device-map auto + --max-cpu-mem + --offload-folder 走磁盘 offload
    if device_map_arg:
        device_map = device_map_arg
    else:
        device_map = "auto" if (device.startswith("cuda") and torch.cuda.is_available()) else None

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_type = getattr(cfg, "model_type", "").lower()

    extra_kwargs: Dict[str, Any] = dict(ignore_mismatched_sizes=True)
    max_memory = _parse_max_memory(max_cpu_mem)

    # 如果要磁盘 offload，transformers 需要 offload_folder / max_memory 等参数
    if device_map == "auto" and max_memory is not None:
        if not offload_folder:
            offload_folder = "./.offload"
        extra_kwargs.update(
            dict(
                max_memory=max_memory,
                offload_folder=offload_folder,
                offload_state_dict=True,
            )
        )

    if "qwen3_vl" in model_type or cfg.__class__.__name__ in ("Qwen3VLConfig", "Qwen3VLMoeConfig"):
        if ("moe" in model_type) and (Qwen3VLMoeForConditionalGeneration is not None):
            model = _call_from_pretrained(
                Qwen3VLMoeForConditionalGeneration,
                model_path,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                **extra_kwargs,
            )
        elif Qwen3VLForConditionalGeneration is not None:
            model = _call_from_pretrained(
                Qwen3VLForConditionalGeneration,
                model_path,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                **extra_kwargs,
            )
        else:
            raise RuntimeError(
                "Detected Qwen3-VL config but transformers doesn't expose Qwen3VLForConditionalGeneration. "
                "Please upgrade transformers."
            )
    else:
        if AutoModelForVision2Seq is not None:
            model = _call_from_pretrained(
                AutoModelForVision2Seq,
                model_path,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                **extra_kwargs,
            )
        else:
            model = _call_from_pretrained(
                AutoModelForCausalLM,
                model_path,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                **extra_kwargs,
            )

    if device_map is None:
        model = model.to(device)

    model.eval()
    return processor, model


class CEDExperiment:
    def __init__(self, model_path: str, device: str, dtype: str, trust_remote_code: bool,
                 device_map_arg: Optional[str], max_cpu_mem: Optional[str], offload_folder: Optional[str]):
        self.processor, self.model = load_model(
            model_path,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            device_map_arg=device_map_arg,
            max_cpu_mem=max_cpu_mem,
            offload_folder=offload_folder,
        )
        self.num_layers = _get_num_hidden_layers(self.model.config, self.model)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", type=str, default="auto")
    p.add_argument("--local-dir", type=str, default="../dataprepare/models/Qwen3-VL-8B-Instruct")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--preferred-root", action="append", default=[])
    p.add_argument("--fallback-repo", type=str, default="Qwen/Qwen3-VL-8B-Instruct")

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--download-only", action="store_true")

    # ✅ CPU 低内存加载（可选）
    p.add_argument("--device-map", type=str, default=None, help="e.g. auto")
    p.add_argument("--max-cpu-mem", type=str, default=None, help="e.g. 24GiB (used with --device-map auto)")
    p.add_argument("--offload-folder", type=str, default=None, help="folder for disk offload")

    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    preferred_roots: List[str] = []
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

    exp = CEDExperiment(
        model_path,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device_map_arg=args.device_map,
        max_cpu_mem=args.max_cpu_mem,
        offload_folder=args.offload_folder,
    )
    print("[OK] Loaded model. num_layers = {}".format(exp.num_layers), flush=True)

    if args.smoke_test:
        import torch
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello World"}]}]
        inputs = exp.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(exp.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = exp.model.generate(**inputs, max_new_tokens=16)
        in_len = inputs["input_ids"].shape[-1]
        trimmed = generated_ids[:, in_len:]
        out_text = exp.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print("[SMOKE_TEST_OUTPUT]", out_text, flush=True)


if __name__ == "__main__":
    main()
