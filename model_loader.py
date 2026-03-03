"""model_loader.py — 加载 Qwen3-VL 模型，仅保留必要逻辑。"""

from __future__ import annotations
import os
import sys
from typing import Tuple, Any

import torch


def load_qwen3vl(
    model_dir: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
) -> Tuple[Any, Any, Any]:
    """
    返回 (processor, model, config)。
    model 已 eval() 且放到指定 device。
    """
    from transformers import AutoProcessor, AutoConfig

    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    # ---------- dtype ----------
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": torch.bfloat16,  # H200 默认 bf16
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # ---------- config → 选模型类 ----------
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    model_type = getattr(cfg, "model_type", "").lower()

    # Qwen3-VL 专用类
    model_cls = None
    if "qwen3_vl" in model_type:
        if "moe" in model_type:
            try:
                from transformers import Qwen3VLMoeForConditionalGeneration
                model_cls = Qwen3VLMoeForConditionalGeneration
            except ImportError:
                pass
        if model_cls is None:
            try:
                from transformers import Qwen3VLForConditionalGeneration
                model_cls = Qwen3VLForConditionalGeneration
            except ImportError:
                pass
    if model_cls is None:
        try:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq
        except ImportError:
            from transformers import AutoModelForCausalLM
            model_cls = AutoModelForCausalLM

    # ---------- 加载 ----------
    device_map = "auto" if device.startswith("cuda") and torch.cuda.is_available() else None
    try:
        model = model_cls.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        # 某些版本用 dtype 而非 torch_dtype
        model = model_cls.from_pretrained(
            model_dir,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

    if device_map is None:
        model = model.to(device)

    model.eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=trust_remote_code)

    return processor, model, cfg


def get_num_layers(cfg, model=None) -> int:
    """从 config 或模型结构推断 transformer 层数。"""
    for attr in ["num_hidden_layers"]:
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    for sub_name in ["text_config", "llm_config", "language_config"]:
        sub = getattr(cfg, sub_name, None)
        if sub and hasattr(sub, "num_hidden_layers"):
            return int(sub.num_hidden_layers)
    if model is not None:
        for accessor in [
            lambda m: len(m.model.layers),
            lambda m: len(m.model.model.layers),
        ]:
            try:
                return int(accessor(model))
            except Exception:
                pass
    raise AttributeError("Cannot determine num_hidden_layers")
