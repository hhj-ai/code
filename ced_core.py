"""CED 核心：hook + token 均值替换 + JS/熵计算。

流程：正常前向 → 替换目标区域 visual token 为周围均值 → 替换后前向 → 比较。
CED = JS(P_orig || P_replaced) + λ_e · max(0, H(P_replaced) - H(P_orig))
"""

from __future__ import annotations
from typing import Dict, List, Any
from contextlib import contextmanager

import torch
import torch.nn.functional as F


# ─────── 度量 ───────

def js_div(p, q, eps=1e-12):
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p / m).log()).sum(-1) + (q * (q / m).log()).sum(-1))

def ent(p, eps=1e-12):
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(-1)

def kl_div(p, q, eps=1e-12):
    p, q = p.clamp_min(eps), q.clamp_min(eps)
    return (p * (p / q).log()).sum(-1)

def cos_dist(a, b):
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


# ─────── Hook：捕获 Hidden States ───────

class HiddenCapture:
    def __init__(self):
        self.data: Dict[int, torch.Tensor] = {}
        self._hooks = []

    def register(self, model, layer_ids):
        layers = model.model.layers  # Qwen3-VL 结构
        for i in layer_ids:
            if i < len(layers):
                self._hooks.append(layers[i].register_forward_hook(
                    lambda mod, inp, out, idx=i: self.data.update(
                        {idx: (out[0] if isinstance(out, tuple) else out).detach()}
                    )
                ))

    def clear(self):   self.data.clear()
    def remove(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()


# ─────── Hook：Embedding 层 Token 替换 ───────

class TokenReplacer:
    def __init__(self):
        self._hook = None
        self.target: List[int] = []
        self.surround: List[int] = []
        self.active = False

    def register(self, model):
        embed = model.model.embed_tokens  # Qwen3-VL 结构
        self._hook = embed.register_forward_hook(self._fn)

    def _fn(self, mod, inp, out):
        if not self.active or not self.target or not self.surround:
            return out
        x = out.clone()
        sur = [p for p in self.surround if p < x.shape[1]]
        if not sur: return out
        mean_v = x[:, sur, :].mean(dim=1, keepdim=True)
        for p in self.target:
            if p < x.shape[1]:
                x[:, p, :] = mean_v.squeeze(1)
        return x

    def set(self, target, surround):
        self.target, self.surround = target, surround

    def remove(self):
        if self._hook: self._hook.remove(); self._hook = None


@contextmanager
def replacing(rep: TokenReplacer):
    rep.active = True
    try: yield
    finally: rep.active = False


# ─────── CED 计算器 ───────

class CEDComputer:
    def __init__(self, model, processor, layers, device="cuda:0"):
        self.model, self.processor, self.device = model, processor, device
        self.layers = layers
        self.rep = TokenReplacer(); self.rep.register(model)
        self.cap1 = HiddenCapture(); self.cap1.register(model, layers)
        self.cap2 = HiddenCapture(); self.cap2.register(model, layers)

    def cleanup(self):
        self.rep.remove(); self.cap1.remove(); self.cap2.remove()

    @torch.no_grad()
    def compute(self, inputs, tgt_abs, sur_abs, lambdas=(0.0, 0.1, 0.2)):
        if not tgt_abs: return {"error": "no_target"}

        # Pass 1: 原始
        self.cap1.clear(); self.rep.active = False
        o1 = self.model(**inputs)
        lg1 = o1.logits[:, -1, :].float(); p1 = F.softmax(lg1, dim=-1)
        hs1 = {k: v.clone() for k, v in self.cap1.data.items()}

        # Pass 2: 替换
        self.cap2.clear(); self.rep.set(tgt_abs, sur_abs)
        with replacing(self.rep):
            o2 = self.model(**inputs)
        lg2 = o2.logits[:, -1, :].float(); p2 = F.softmax(lg2, dim=-1)
        hs2 = {k: v.clone() for k, v in self.cap2.data.items()}

        # 指标
        r = {}
        js = js_div(p1, p2).item(); h1 = ent(p1).item(); h2 = ent(p2).item()
        r["logits_js"] = js
        r["logits_entropy_orig"] = h1
        r["logits_entropy_replaced"] = h2
        r["logits_entropy_delta"] = h2 - h1
        r["logits_kl"] = kl_div(p1, p2).item()
        r["logits_cosine_dist"] = cos_dist(lg1, lg2).item()
        penalty = max(0.0, h2 - h1)
        r["entropy_penalty_only"] = penalty
        for lam in lambdas:
            r[f"ced_lambda_{lam:.2f}"] = js + lam * penalty

        # 中间层
        for i in self.layers:
            if i in hs1 and i in hs2:
                a, b = hs1[i][:, -1, :].float(), hs2[i][:, -1, :].float()
                r[f"layer_{i}_js"] = js_div(F.softmax(a, -1), F.softmax(b, -1)).item()
                r[f"layer_{i}_cosine"] = cos_dist(a, b).item()
        return r

    @torch.no_grad()
    def generate(self, inputs, max_new=32) -> str:
        self.rep.active = False
        g = self.model.generate(**inputs, max_new_tokens=max_new)
        return self.processor.batch_decode(
            g[:, inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]


# ─────── 工具函数 ───────

def prepare_inputs(processor, image, question, device):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)
    return {k: v.to(device) for k, v in inputs.items()}


def get_image_token_id(processor) -> int:
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    for name in ["<|image_pad|>", "<|vision_pad|>", "<image>"]:
        tid = tok.convert_tokens_to_ids(name)
        if tid != tok.unk_token_id:
            return tid
    assert hasattr(tok, "image_token_id"), "找不到 image token id"
    return tok.image_token_id
