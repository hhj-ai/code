"""模型加载。Qwen3-VL 专用，失败直接报错。"""

import os, torch
from transformers import AutoProcessor, AutoConfig, Qwen3VLForConditionalGeneration


def load(model_dir: str, device="cuda:0", dtype="bfloat16"):
    """返回 (processor, model, config)。"""
    assert os.path.isfile(f"{model_dir}/config.json"), f"模型不存在: {model_dir}"

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}.get(dtype, torch.bfloat16)

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    return processor, model, cfg


def num_layers(cfg, model=None) -> int:
    for sub in [cfg, getattr(cfg, "text_config", None), getattr(cfg, "llm_config", None)]:
        if sub and hasattr(sub, "num_hidden_layers"):
            return int(sub.num_hidden_layers)
    if model:
        try: return len(model.model.layers)
        except: pass
    raise AttributeError("Cannot determine num_hidden_layers")
