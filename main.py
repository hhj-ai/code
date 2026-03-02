#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""main.py (v9)

你问“为什么 gpu.sh 没有跑实验”：
- 因为当前 gpu.sh 调用的是 `main.py --smoke-test`，只做“能否加载 + 简单生成”验活。
- 这一步是为了先把前面一连串坑（Qwen3-VL 类别、dtype、conda 绝对路径、CUDA so 冲突）全部排干净。
  现在模型能在 GPU 上正常加载并输出了，说明环境层 OK。

✅ v9 增强：支持“定位模型后，继续执行你的实验入口脚本”
- 用法：把你真正的实验命令放在 `--exec -- <cmd...>` 后面
- 你可以在命令里写 `{MODEL_PATH}` 占位符，本脚本会自动替换成解析到的本地模型目录。
  例如：
    python main.py --exec -- python your_experiment.py --model_path {MODEL_PATH} --device cuda:0

另外：
- 保留 v7/v8 的修复：Qwen3-VL 用 Qwen3VLForConditionalGeneration；默认 auto 复用本地；不用 hf CLI；dtype/torch_dtype 兼容。
"""

from __future__ import annotations

import os
import sys
import argparse
import subprocess
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


def load_model(model_path: str, device: str, dtype: str, trust_remote_code: bool):
    torch, AutoProcessor, AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, AutoModelForVision2Seq, AutoModelForCausalLM = _import_transformers()

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    if dtype == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.bfloat16
    else:
        torch_dtype = getattr(torch, dtype)

    device_map = "auto" if (device.startswith("cuda") and torch.cuda.is_available()) else None

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_type = getattr(cfg, "model_type", "").lower()

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


def _substitute_placeholders(argv: List[str], model_path: str) -> List[str]:
    out: List[str] = []
    for x in argv:
        out.append(x.replace("{MODEL_PATH}", model_path))
    return out


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
    p.add_argument("--smoke-test", action="store_true")

    p.add_argument("--exec", dest="exec_mode", action="store_true")
    p.add_argument("exec_argv", nargs=argparse.REMAINDER)

    p.add_argument("--verify-load", action="store_true", help="Load model once before exec for verification.")
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
    model_path = os.path.abspath(model_path)
    print("[OK] Model ready at: {}".format(model_path), flush=True)

    if args.download_only:
        return

    if args.exec_mode:
        if not args.exec_argv:
            raise SystemExit("`--exec` specified but no command provided. Usage: --exec -- python your_exp.py ...")

        exec_argv = args.exec_argv
        if exec_argv and exec_argv[0] == "--":
            exec_argv = exec_argv[1:]
        exec_argv = _substitute_placeholders(exec_argv, model_path)

        if args.verify_load:
            exp = CEDExperiment(model_path, device=args.device, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
            print("[OK] Verified load. num_layers = {}".format(exp.num_layers), flush=True)

        env = os.environ.copy()
        env["CED_MODEL_PATH"] = model_path

        print("[EXEC]", " ".join(exec_argv), flush=True)
        r = subprocess.run(exec_argv, env=env)
        raise SystemExit(r.returncode)

    exp = CEDExperiment(model_path, device=args.device, dtype=args.dtype, trust_remote_code=args.trust_remote_code)
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
