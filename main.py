#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""main.py

目标：
- 仍然保持你要的布局（cpu.sh / gpu.sh / main.py）
- 通过 cpu.sh / gpu.sh 使用 conda 环境的绝对路径 python（不依赖 conda activate）
- 修复/规避：
  1) hf CLI 的 --local-dir-use-symlinks 不兼容：不再使用
  2) Qwen3VLConfig 没有顶层 num_hidden_layers：自动从 text_config/llm_config 等读取，必要时从模型结构推断
  3) visual pos_embed 等权重尺寸不匹配：ignore_mismatched_sizes=True
  4) 允许 --download-only：下载阶段不导入 torch，避免“torch import 直接炸”导致下载也跑不了

兼容：
- Python 3.8+（不使用 list[str] 这类 3.9+ 注解语法）
"""

from __future__ import annotations

import os
import sys
import argparse
import subprocess
from typing import Any, Optional, List, Dict

from huggingface_hub import snapshot_download


def _run(cmd: List[str]) -> None:
    """Run a command; raise on failure."""
    print("[CMD]", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout, flush=True)
    if p.returncode != 0:
        raise RuntimeError("Command failed ({}): {}".format(p.returncode, " ".join(cmd)))


def ensure_downloaded(repo_id_or_path: str, local_dir: str, cache_dir: Optional[str]) -> str:
    """
    Ensure model weights exist locally.

    - If repo_id_or_path is a local directory -> return it.
    - Else try `hf download ... --local-dir ...` (no deprecated flags).
    - If CLI fails, fall back to huggingface_hub.snapshot_download().
    """
    if os.path.isdir(repo_id_or_path):
        return repo_id_or_path

    os.makedirs(local_dir, exist_ok=True)

    def has_weights(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.endswith(".safetensors") or fn.endswith(".bin"):
                    return True
        return False

    if has_weights(local_dir) and os.path.exists(os.path.join(local_dir, "config.json")):
        return local_dir

    # 1) Try hf CLI if available
    try:
        cmd = ["hf", "download", repo_id_or_path, "--repo-type", "model", "--local-dir", local_dir]
        if cache_dir:
            cmd += ["--cache-dir", cache_dir]
        cmd += ["--resume-download"]
        _run(cmd)
        if has_weights(local_dir):
            return local_dir
    except Exception as e:
        print("[WARN] hf CLI download failed; fallback to snapshot_download(): {}".format(e), file=sys.stderr)

    # 2) Python fallback
    kwargs: Dict[str, Any] = dict(
        repo_id=repo_id_or_path,
        repo_type="model",
        local_dir=local_dir,
    )
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    # local_dir_use_symlinks 在不同版本可能被改动；安全处理
    try:
        snapshot_download(local_dir_use_symlinks=False, **kwargs)  # type: ignore
    except TypeError:
        snapshot_download(**kwargs)

    if not has_weights(local_dir):
        raise RuntimeError("Download finished but no weights found in: {}".format(local_dir))
    return local_dir


def _get_num_hidden_layers(cfg: Any, model: Optional[Any] = None) -> int:
    """Robustly determine transformer depth across common config layouts."""
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
    """
    Delay-import torch/transformers so --download-only can run even if torch import fails.
    If torch import fails, print a crisp diagnostic and re-raise.
    """
    try:
        import torch  # noqa
    except Exception as e:
        msg = (
            "\n[FATAL] import torch 失败。\n"
            "这通常不是脚本问题，而是 CUDA 动态库版本不匹配（例如 libcusparse.so.12 需要更新的 libnvJitLink.so.12）。\n"
            "你可以：\n"
            "  1) 优先用 cpu.sh/gpu.sh 里设置的 LD_LIBRARY_PATH，让 conda env 自带的 nvidia 库优先生效；\n"
            "  2) 或者重装与驱动兼容的 PyTorch CUDA 版本；\n"
            "  3) 如果只想在 CPU 跑，安装 CPU-only 的 torch。\n"
            f"原始异常: {repr(e)}\n"
        )
        print(msg, file=sys.stderr)
        raise

    # transformers imports
    from transformers import AutoProcessor  # noqa
    try:
        from transformers import AutoModelForVision2Seq  # noqa
        has_v2s = True
    except Exception:
        has_v2s = False

    from transformers import AutoModelForCausalLM  # noqa
    return torch, AutoProcessor, has_v2s, AutoModelForCausalLM, (AutoModelForVision2Seq if has_v2s else None)


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
        self.device = device
        self.processor, self.model = load_model(model_path, device=device, dtype=dtype, trust_remote_code=trust_remote_code)
        self.num_layers = _get_num_hidden_layers(self.model.config, self.model)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--local-dir", type=str, default="./models/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="auto")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = ensure_downloaded(args.repo_id, args.local_dir, args.cache_dir)
    print("[OK] Model files ready at: {}".format(model_path), flush=True)

    if args.download_only:
        return

    exp = CEDExperiment(model_path, device=args.device, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
    print("[OK] Loaded model. num_layers = {}".format(exp.num_layers), flush=True)

    if args.smoke_test:
        # Small smoke test (text only)
        import torch
        with torch.no_grad():
            inputs = exp.processor(text="Hello World", return_tensors="pt")
            inputs = {k: v.to(exp.model.device) for k, v in inputs.items()}
            out = exp.model.generate(**inputs, max_new_tokens=16)
            txt = exp.processor.decode(out[0], skip_special_tokens=True)
            print("[SMOKE_TEST_OUTPUT]", txt, flush=True)


if __name__ == "__main__":
    main()
