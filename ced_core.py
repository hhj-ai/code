"""ced_core.py — CED (Counterfactual Entropy Divergence) 核心计算。

核心流程：
1. 正常前向传播 → 获取 P_orig（output logits + 各层 hidden states）
2. 在 visual token 输入层做目标区域的 token 均值替换
3. 替换后前向传播 → 获取 P_replaced
4. CED = JS(P_orig || P_replaced) + λ_e · max(0, H(P_replaced) - H(P_orig))

多层分析：在第1、2步都 hook 中间层 hidden states，计算各层的 JS。
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import numpy as np


# ─────────────────── 基础度量 ───────────────────

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Jensen-Shannon divergence between two probability distributions.
    p, q: [..., vocab_size] 概率分布（已 softmax）
    返回: [...] 标量
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy. p: [..., vocab_size] 概率分布。"""
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(p || q)"""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p / q).log()).sum(dim=-1)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity. a, b: [..., dim]"""
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


# ─────────────────── Hook 机制 ───────────────────

class HiddenStateCapture:
    """Hook 到 LLM 指定层，捕获 hidden states。"""

    def __init__(self):
        self.captured: Dict[int, torch.Tensor] = {}
        self._hooks = []

    def register(self, model: torch.nn.Module, layer_indices: List[int]):
        """注册 hook 到指定层。
        
        兼容 Qwen3-VL 的模型结构：
        model.model.layers[i]  或  model.language_model.model.layers[i]
        """
        layers = self._get_layers(model)
        for idx in layer_indices:
            if idx >= len(layers):
                print(f"[WARN] Layer {idx} >= total {len(layers)}, skipping")
                continue
            hook = layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def _get_layers(self, model) -> torch.nn.ModuleList:
        """获取 LLM 的 transformer 层列表。"""
        for path in [
            lambda m: m.model.layers,
            lambda m: m.model.model.layers,
            lambda m: m.language_model.model.layers,
        ]:
            try:
                layers = path(model)
                if isinstance(layers, torch.nn.ModuleList):
                    return layers
            except (AttributeError, TypeError):
                continue
        raise AttributeError("Cannot find transformer layers in model")

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            # output 通常是 tuple，第一个元素是 hidden states
            if isinstance(output, tuple):
                hs = output[0]
            else:
                hs = output
            self.captured[layer_idx] = hs.detach()
        return hook_fn

    def clear(self):
        self.captured.clear()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class VisualTokenReplacer:
    """在 LLM 输入的 embedding 层做 visual token 替换。
    
    原理：Hook 到 LLM 的 embedding 输出（即进入第一层 transformer 之前），
    将目标区域的 visual token 替换为周围 token 均值。
    """

    def __init__(self):
        self._hook = None
        self.target_positions: List[int] = []
        self.surround_positions: List[int] = []
        self.active = False

    def register(self, model: torch.nn.Module):
        """Hook 到 embed_tokens 或等效的 embedding 层。"""
        embed_layer = self._get_embed_layer(model)
        self._hook = embed_layer.register_forward_hook(self._replace_hook)

    def _get_embed_layer(self, model) -> torch.nn.Module:
        for path in [
            lambda m: m.model.embed_tokens,
            lambda m: m.model.model.embed_tokens,
            lambda m: m.language_model.model.embed_tokens,
        ]:
            try:
                layer = path(model)
                if isinstance(layer, torch.nn.Module):
                    return layer
            except (AttributeError, TypeError):
                continue
        raise AttributeError("Cannot find embed_tokens layer")

    def _replace_hook(self, module, input, output):
        """在 embedding 输出上做替换。"""
        if not self.active:
            return output
        if not self.target_positions or not self.surround_positions:
            return output

        # output: [batch, seq_len, hidden_dim]
        modified = output.clone()

        # 计算周围 token 的均值
        surround_indices = [p for p in self.surround_positions if p < modified.shape[1]]
        if not surround_indices:
            return output

        surround_mean = modified[:, surround_indices, :].mean(dim=1, keepdim=True)

        # 替换目标位置
        for pos in self.target_positions:
            if pos < modified.shape[1]:
                modified[:, pos, :] = surround_mean.squeeze(1)

        return modified

    def set_replacement(self, target_pos: List[int], surround_pos: List[int]):
        self.target_positions = target_pos
        self.surround_positions = surround_pos

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


# ─────────────────── CED 计算 ───────────────────

@contextmanager
def replacement_context(replacer: VisualTokenReplacer):
    """上下文管理器：进入时激活替换，退出时关闭。"""
    replacer.active = True
    try:
        yield
    finally:
        replacer.active = False


class CEDComputer:
    """CED 计算器：封装完整的 CED 测量流程。"""

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        monitor_layers: List[int],
        device: str = "cuda:0",
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.monitor_layers = monitor_layers

        # 初始化 hook
        self.replacer = VisualTokenReplacer()
        self.replacer.register(model)

        self.capture_orig = HiddenStateCapture()
        self.capture_orig.register(model, monitor_layers)

        self.capture_replaced = HiddenStateCapture()
        self.capture_replaced.register(model, monitor_layers)

    def cleanup(self):
        self.replacer.remove_hook()
        self.capture_orig.remove_hooks()
        self.capture_replaced.remove_hooks()

    @torch.no_grad()
    def compute_ced(
        self,
        inputs: Dict[str, torch.Tensor],
        target_abs_positions: List[int],
        surround_abs_positions: List[int],
        lambda_e_values: List[float] = [0.0, 0.1, 0.2],
    ) -> Dict[str, Any]:
        """计算完整的 CED 指标。

        Args:
            inputs: processor 输出的 tokenized inputs（已移到 device）
            target_abs_positions: 目标 token 在 input_ids 中的绝对位置
            surround_abs_positions: 周围 token 在 input_ids 中的绝对位置
            lambda_e_values: 熵正则系数扫描值

        Returns:
            dict 包含：
            - logits_js: logits 层的 JS 散度
            - logits_entropy_orig/replaced: logits 层的熵
            - ced_{lambda}: 各 lambda 下的 CED 值
            - layer_{i}_js: 各中间层的 JS 散度
            - 各种消融指标（KL、cosine distance 等）
        """
        if not target_abs_positions:
            return {"error": "no_target_positions"}

        # ---- Pass 1: 正常前向 ----
        self.capture_orig.clear()
        self.replacer.active = False
        out_orig = self.model(**inputs)
        logits_orig = out_orig.logits[:, -1, :].float()  # 最后一个 token 的 logits
        probs_orig = F.softmax(logits_orig, dim=-1)

        # 保存各层 hidden states
        hs_orig = {k: v.clone() for k, v in self.capture_orig.captured.items()}

        # ---- Pass 2: 替换后前向 ----
        self.capture_replaced.clear()
        self.replacer.set_replacement(target_abs_positions, surround_abs_positions)
        with replacement_context(self.replacer):
            out_replaced = self.model(**inputs)
        logits_replaced = out_replaced.logits[:, -1, :].float()
        probs_replaced = F.softmax(logits_replaced, dim=-1)

        hs_replaced = {k: v.clone() for k, v in self.capture_replaced.captured.items()}

        # ---- 计算指标 ----
        results = {}

        # Logits 层
        js_val = js_divergence(probs_orig, probs_replaced).item()
        h_orig = entropy(probs_orig).item()
        h_replaced = entropy(probs_replaced).item()
        kl_val = kl_divergence(probs_orig, probs_replaced).item()

        results["logits_js"] = js_val
        results["logits_entropy_orig"] = h_orig
        results["logits_entropy_replaced"] = h_replaced
        results["logits_entropy_delta"] = h_replaced - h_orig
        results["logits_kl"] = kl_val

        # CED = JS + λ_e · max(0, H_replaced - H_orig)
        entropy_penalty = max(0.0, h_replaced - h_orig)
        for lam in lambda_e_values:
            results[f"ced_lambda_{lam:.2f}"] = js_val + lam * entropy_penalty

        # 纯熵惩罚（消融用）
        results["entropy_penalty_only"] = entropy_penalty

        # Logits cosine distance（消融用）
        cos_dist = cosine_distance(logits_orig, logits_replaced).item()
        results["logits_cosine_dist"] = cos_dist

        # 中间层 JS 散度
        for layer_idx in self.monitor_layers:
            if layer_idx in hs_orig and layer_idx in hs_replaced:
                # 取最后一个 token 的 hidden state
                h_o = hs_orig[layer_idx][:, -1, :].float()
                h_r = hs_replaced[layer_idx][:, -1, :].float()

                # 对 hidden states 做 softmax 得到伪分布，再算 JS
                p_o = F.softmax(h_o, dim=-1)
                p_r = F.softmax(h_r, dim=-1)
                layer_js = js_divergence(p_o, p_r).item()
                layer_cos = cosine_distance(h_o, h_r).item()

                results[f"layer_{layer_idx}_js"] = layer_js
                results[f"layer_{layer_idx}_cosine"] = layer_cos

        return results

    @torch.no_grad()
    def generate_answer(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int = 32,
    ) -> str:
        """生成回答文本。"""
        gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        in_len = inputs["input_ids"].shape[-1]
        trimmed = gen_ids[:, in_len:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


def prepare_inputs(
    processor: Any,
    image,
    question: str,
    device: str,
) -> Dict[str, torch.Tensor]:
    """将 image + question 转换为模型输入。"""
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}
    ]
    # Qwen3-VL 的 processor 需要特殊处理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 使用 processor 处理图像和文本
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def get_image_token_id(processor) -> int:
    """获取 image placeholder token 的 id。"""
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Qwen3-VL 使用 <|image_pad|> 或 <|vision_pad|>
    for token_name in ["<|image_pad|>", "<|vision_pad|>", "<image>"]:
        token_id = tokenizer.convert_tokens_to_ids(token_name)
        if token_id != tokenizer.unk_token_id:
            return token_id

    # 尝试从 config 获取
    if hasattr(tokenizer, "image_token_id"):
        return tokenizer.image_token_id

    raise ValueError("Cannot find image token id in processor/tokenizer")
