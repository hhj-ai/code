#!/usr/bin/env python3
"""
CED Phase 0: Counterfactual Evidence Divergence Signal Validation
=================================================================
Validates that replacing target-region visual tokens with surrounding-mean
produces a JS-divergence signal that distinguishes "genuinely seeing" from
"hallucinating based on text priors".

Experiments:
  P0-a  Architecture probing  (hook positions, merger ratio, end-to-end check)
  P0-b  CED signal validation (behavior grouping, formula ablation,
        cross-layer analysis, cross-task consistency, AUC-ROC)

Usage:
  python main.py --mode probe    ...   # P0-a only
  python main.py --mode validate ...   # P0-b full suite
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ced_p0")


# ============================================================================
# 1. Math utilities
# ============================================================================

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """Compute Jensen-Shannon divergence between two probability distributions.
    
    Args:
        p, q: 1-D tensors (probability distributions, should sum to 1).
    Returns:
        JS divergence (float, in [0, ln2] for natural log).
    """
    p = p.float().clamp(min=eps)
    q = q.float().clamp(min=eps)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = F.kl_div(m.log(), p, reduction="sum", log_target=False)
    kl_qm = F.kl_div(m.log(), q, reduction="sum", log_target=False)
    return (0.5 * (kl_pm + kl_qm)).item()


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> float:
    """KL(P || Q)."""
    p = p.float().clamp(min=eps)
    q = q.float().clamp(min=eps)
    p = p / p.sum()
    q = q / q.sum()
    return F.kl_div(q.log(), p, reduction="sum", log_target=False).item()


def entropy(p: torch.Tensor, eps: float = 1e-10) -> float:
    """Shannon entropy of a distribution."""
    p = p.float().clamp(min=eps)
    p = p / p.sum()
    return -(p * p.log()).sum().item()


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine distance = 1 - cosine_similarity."""
    return (1.0 - F.cosine_similarity(a.float().unsqueeze(0),
                                       b.float().unsqueeze(0)).item())


def compute_ced(js_val: float, ent_orig: float, ent_repl: float,
                lambda_e: float) -> float:
    """CED = JS + lambda_e * max(0, entropy_replaced - entropy_original).
    
    The entropy penalty discourages replacement that merely adds noise
    (which increases entropy but not true information difference).
    """
    return js_val + lambda_e * max(0.0, ent_repl - ent_orig)


# ============================================================================
# 2. Visual token mapping
# ============================================================================

def bbox_to_merged_token_indices(
    bbox: list,
    image_size: tuple,
    resized_size: tuple,
    patch_size: int = 14,
    merge_size: int = 2,
) -> list:
    """Map a bounding box (in original pixel coords) to merged visual token
    indices after ViT patching + spatial merge.

    Args:
        bbox:         [x, y, w, h] in original image coordinates (COCO format).
        image_size:   (W_orig, H_orig) original image size.
        resized_size: (W_resized, H_resized) actual size fed to ViT.
        patch_size:   ViT patch side length (default 14 for Qwen2.5-VL).
        merge_size:   spatial merge factor (default 2 → 2×2 merge).

    Returns:
        List of linear indices into the merged visual token sequence.
    """
    x, y, w, h = bbox
    W_orig, H_orig = image_size
    W_res, H_res = resized_size

    # Scale bbox to resized image
    sx = W_res / W_orig
    sy = H_res / H_orig
    x1, y1 = x * sx, y * sy
    x2, y2 = (x + w) * sx, (y + h) * sy

    effective_patch = patch_size * merge_size  # 28 for default
    grid_w = W_res // effective_patch
    grid_h = H_res // effective_patch

    # Token indices (inclusive)
    col_start = max(0, int(x1 // effective_patch))
    col_end   = min(grid_w - 1, int(x2 // effective_patch))
    row_start = max(0, int(y1 // effective_patch))
    row_end   = min(grid_h - 1, int(y2 // effective_patch))

    indices = []
    for r in range(row_start, row_end + 1):
        for c in range(col_start, col_end + 1):
            indices.append(r * grid_w + c)
    return indices


def get_resized_image_size(image: Image.Image, min_pixels: int = 3136,
                           max_pixels: int = 12845056,
                           patch_size: int = 14, merge_size: int = 2) -> tuple:
    """Compute the resized image dimensions that Qwen2.5-VL processor uses.
    
    Qwen2.5-VL resizes images so that total pixels ∈ [min_pixels, max_pixels]
    and both dimensions are divisible by (patch_size * merge_size = 28).
    """
    W, H = image.size
    eff = patch_size * merge_size  # 28

    # Scale to fit within pixel budget
    total = W * H
    if total > max_pixels:
        scale = math.sqrt(max_pixels / total)
        W, H = int(W * scale), int(H * scale)
    if total < min_pixels:
        scale = math.sqrt(min_pixels / total)
        W, H = int(W * scale), int(H * scale)

    # Round to nearest multiple of 28
    W = max(eff, round(W / eff) * eff)
    H = max(eff, round(H / eff) * eff)
    return W, H


# ============================================================================
# 3. COCO data preparation
# ============================================================================

# Common objects to use as negative (absent) categories
COMMON_CATEGORIES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat",
    "horse", "cow", "elephant", "bear", "giraffe", "chair", "couch",
    "dining table", "laptop", "cell phone", "book", "clock", "vase",
    "cup", "bottle", "bowl", "banana", "apple", "sandwich", "pizza",
]


def prepare_samples(coco: COCO, img_dir: str, num_samples: int = 500,
                    seed: int = 42) -> dict:
    """Prepare samples for four task types.

    Returns dict with keys: existence, spatial, attribute, counting.
    Each value is a list of sample dicts.
    """
    rng = random.Random(seed)
    cat_name_to_id = {c["name"]: c["id"] for c in coco.cats.values()}
    cat_id_to_name = {c["id"]: c["name"] for c in coco.cats.values()}

    all_img_ids = list(coco.imgs.keys())
    rng.shuffle(all_img_ids)

    samples = {"existence": [], "spatial": [], "attribute": [], "counting": []}
    target_per_type = num_samples // 4

    for img_id in tqdm(all_img_ids, desc="Preparing samples", leave=False):
        if all(len(v) >= target_per_type for v in samples.values()):
            break

        img_info = coco.imgs[img_id]
        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.isfile(img_path):
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            continue

        W_orig, H_orig = img_info["width"], img_info["height"]
        present_cats = set(cat_id_to_name.get(a["category_id"], "") for a in anns)

        # --- Existence (positive + negative) ---
        if len(samples["existence"]) < target_per_type:
            for ann in anns[:2]:
                cat = cat_id_to_name.get(ann["category_id"], "")
                if not cat or ann["bbox"][2] < 20 or ann["bbox"][3] < 20:
                    continue
                # Positive: object is in image
                samples["existence"].append({
                    "img_id": img_id, "img_path": img_path,
                    "image_size": (W_orig, H_orig),
                    "question": f"Is there a {cat} in this image? Answer yes or no.",
                    "answer_gt": "yes", "target_bbox": ann["bbox"],
                    "category": cat, "task": "existence",
                })
            # Negative: object is NOT in image
            absent = [c for c in COMMON_CATEGORIES if c not in present_cats]
            if absent:
                neg_cat = rng.choice(absent)
                samples["existence"].append({
                    "img_id": img_id, "img_path": img_path,
                    "image_size": (W_orig, H_orig),
                    "question": f"Is there a {neg_cat} in this image? Answer yes or no.",
                    "answer_gt": "no", "target_bbox": None,
                    "category": neg_cat, "task": "existence",
                })

        # --- Spatial ---
        if len(samples["spatial"]) < target_per_type and len(anns) >= 2:
            a1, a2 = anns[0], anns[1]
            c1 = cat_id_to_name.get(a1["category_id"], "object")
            c2 = cat_id_to_name.get(a2["category_id"], "object")
            cx1 = a1["bbox"][0] + a1["bbox"][2] / 2
            cx2 = a2["bbox"][0] + a2["bbox"][2] / 2
            if abs(cx1 - cx2) > 30:
                rel = "left" if cx1 < cx2 else "right"
                samples["spatial"].append({
                    "img_id": img_id, "img_path": img_path,
                    "image_size": (W_orig, H_orig),
                    "question": (f"Is the {c1} to the {rel} of the {c2}? "
                                 f"Answer yes or no."),
                    "answer_gt": "yes",
                    "target_bbox": a1["bbox"], "category": c1, "task": "spatial",
                })
                # Create a wrong-relation question
                wrong_rel = "right" if rel == "left" else "left"
                samples["spatial"].append({
                    "img_id": img_id, "img_path": img_path,
                    "image_size": (W_orig, H_orig),
                    "question": (f"Is the {c1} to the {wrong_rel} of the {c2}? "
                                 f"Answer yes or no."),
                    "answer_gt": "no",
                    "target_bbox": a1["bbox"], "category": c1, "task": "spatial",
                })

        # --- Attribute (size) ---
        if len(samples["attribute"]) < target_per_type:
            for ann in anns[:1]:
                cat = cat_id_to_name.get(ann["category_id"], "object")
                area = ann["bbox"][2] * ann["bbox"][3]
                img_area = W_orig * H_orig
                ratio = area / img_area
                if ratio > 0.15:
                    size_gt = "large"
                elif ratio > 0.03:
                    size_gt = "medium"
                else:
                    size_gt = "small"
                samples["attribute"].append({
                    "img_id": img_id, "img_path": img_path,
                    "image_size": (W_orig, H_orig),
                    "question": (f"What is the relative size of the {cat} in "
                                 f"this image? Answer large, medium, or small."),
                    "answer_gt": size_gt,
                    "target_bbox": ann["bbox"], "category": cat,
                    "task": "attribute",
                })

        # --- Counting ---
        if len(samples["counting"]) < target_per_type:
            cat_counts = defaultdict(list)
            for ann in anns:
                cat = cat_id_to_name.get(ann["category_id"], "")
                if cat:
                    cat_counts[cat].append(ann)
            for cat, cat_anns in cat_counts.items():
                if 1 <= len(cat_anns) <= 8:
                    # Use first annotation bbox as target
                    samples["counting"].append({
                        "img_id": img_id, "img_path": img_path,
                        "image_size": (W_orig, H_orig),
                        "question": f"How many {cat}s are in this image? Answer with a number.",
                        "answer_gt": str(len(cat_anns)),
                        "target_bbox": cat_anns[0]["bbox"],
                        "category": cat, "task": "counting",
                    })
                    break

    # Truncate to requested size
    for k in samples:
        samples[k] = samples[k][:target_per_type]

    total = sum(len(v) for v in samples.values())
    log.info(f"Prepared {total} samples: " +
             ", ".join(f"{k}={len(v)}" for k, v in samples.items()))
    return samples


# ============================================================================
# 4. CED Experiment engine
# ============================================================================

class CEDExperiment:
    """Handles model loading, visual token replacement, and CED computation."""

    def __init__(self, model_path: str, device: str = "cuda:0",
                 dtype=torch.bfloat16):
        log.info(f"Loading model from {model_path} ...")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        self.device = torch.device(device)
        self.dtype = dtype

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

        # Token IDs
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            "<|image_pad|>"
        )

        # Hook storage for intermediate layer hidden states
        self._layer_states_orig = {}
        self._layer_states_repl = {}
        self._hooks = []
        self._capture_mode = "orig"  # or "repl"

        # Model info
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        log.info(f"Model loaded. Layers={self.num_layers}, "
                 f"Hidden={self.hidden_size}, Dtype={dtype}")

    # ---- Hook management ----

    def register_layer_hooks(self, layer_indices: list):
        """Register forward hooks on specified transformer layers."""
        self.remove_hooks()
        for idx in layer_indices:
            if idx == "logits":
                continue  # logits are captured from model output
            idx = int(idx)
            if idx >= self.num_layers:
                log.warning(f"Layer {idx} >= num_layers {self.num_layers}, skipping")
                continue

            def make_hook(layer_idx):
                def hook_fn(module, args, output):
                    # output: tuple(hidden_states, ...) or just hidden_states
                    hs = output[0] if isinstance(output, tuple) else output
                    store = (self._layer_states_orig
                             if self._capture_mode == "orig"
                             else self._layer_states_repl)
                    # Capture last token hidden state
                    store[layer_idx] = hs[:, -1, :].detach().clone()
                return hook_fn

            h = self.model.model.layers[idx].register_forward_hook(make_hook(idx))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ---- Visual feature replacement via hook ----

    def _setup_visual_hook(self):
        """Hook on model.visual to optionally replace target visual tokens."""
        self._visual_replace_cfg = {
            "active": False,
            "target_indices": [],
            "replacement": None,
        }

        def visual_hook(module, input, output):
            cfg = self._visual_replace_cfg
            if cfg["active"] and cfg["replacement"] is not None:
                modified = output.clone()
                for idx in cfg["target_indices"]:
                    if idx < modified.shape[0]:
                        modified[idx] = cfg["replacement"].to(modified.dtype)
                return modified
            return output

        self._visual_hook_handle = self.model.visual.register_forward_hook(
            visual_hook
        )

    def _remove_visual_hook(self):
        if hasattr(self, "_visual_hook_handle"):
            self._visual_hook_handle.remove()

    # ---- Architecture probing (P0-a) ----

    def probe_architecture(self, sample_image_path: str):
        """P0-a: Print model architecture details and verify hooks."""
        log.info("=" * 60)
        log.info("P0-a: Architecture Probing")
        log.info("=" * 60)

        results = {}

        # 1. Visual encoder structure
        visual = self.model.visual
        log.info(f"Visual encoder type: {type(visual).__name__}")
        log.info(f"  patch_size (from config): {getattr(self.model.config, 'vision_config', {})}")

        # Count visual encoder parameters
        n_vis_params = sum(p.numel() for p in visual.parameters())
        log.info(f"  Visual encoder params: {n_vis_params / 1e6:.1f}M")
        results["visual_params_M"] = round(n_vis_params / 1e6, 1)

        # 2. LLM structure
        log.info(f"LLM layers: {self.num_layers}")
        log.info(f"Hidden size: {self.hidden_size}")
        log.info(f"Vocab size: {self.model.config.vocab_size}")
        results["num_layers"] = self.num_layers
        results["hidden_size"] = self.hidden_size

        # 3. Test with a real image
        log.info(f"\nTesting with image: {sample_image_path}")
        image = Image.open(sample_image_path).convert("RGB")
        W_orig, H_orig = image.size
        W_res, H_res = get_resized_image_size(image)
        log.info(f"  Original size: {W_orig}x{H_orig}")
        log.info(f"  Resized size: {W_res}x{H_res}")
        log.info(f"  Expected merged tokens: {(W_res // 28) * (H_res // 28)}")
        results["test_image_orig_size"] = [W_orig, H_orig]
        results["test_image_resized"] = [W_res, H_res]

        # Process through model
        question = "Describe this image briefly."
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image],
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Count image tokens in input_ids
        input_ids = inputs["input_ids"]
        n_image_tokens = (input_ids == self.image_token_id).sum().item()
        log.info(f"  Image tokens in input_ids: {n_image_tokens}")
        log.info(f"  Total input tokens: {input_ids.shape[1]}")
        results["n_image_tokens"] = n_image_tokens
        results["n_total_tokens"] = input_ids.shape[1]

        # Get visual features directly
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            pv = inputs["pixel_values"].to(dtype=self.dtype)
            grid_thw = inputs.get("image_grid_thw", None)
            if grid_thw is not None:
                grid_thw = grid_thw.to(self.device)
            with torch.no_grad():
                vis_features = self.model.visual(pv, grid_thw=grid_thw)
            log.info(f"  Visual features shape: {vis_features.shape}")
            results["visual_features_shape"] = list(vis_features.shape)

            # Verify merger ratio
            if grid_thw is not None:
                t, h, w = grid_thw[0].tolist()
                log.info(f"  grid_thw: t={t}, h={h}, w={w}")
                expected_pre_merge = int(t * h * w)
                expected_post_merge = vis_features.shape[0]
                ratio = expected_pre_merge / expected_post_merge
                log.info(f"  Pre-merge tokens: {expected_pre_merge}, "
                         f"Post-merge: {expected_post_merge}, "
                         f"Ratio: {ratio:.1f}")
                results["merge_ratio"] = ratio
                results["grid_thw"] = [t, h, w]

        # 4. End-to-end forward test
        log.info("\n  Running end-to-end forward pass...")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        log.info(f"  Output logits shape: {logits.shape}")
        log.info(f"  Last-token logits range: [{logits[0, -1].min():.2f}, "
                 f"{logits[0, -1].max():.2f}]")
        results["logits_shape"] = list(logits.shape)

        # 5. Test visual token replacement
        log.info("\n  Testing visual token replacement...")
        self._setup_visual_hook()
        with torch.no_grad():
            vis_features_check = self.model.visual(pv, grid_thw=grid_thw)
        n_vis = vis_features_check.shape[0]

        # Pick center tokens as target, surrounding as reference
        center = n_vis // 2
        target_idx = list(range(max(0, center - 2), min(n_vis, center + 3)))
        surround_idx = [i for i in range(n_vis) if i not in target_idx]
        replacement = vis_features_check[surround_idx].mean(dim=0)

        self._visual_replace_cfg.update({
            "active": True,
            "target_indices": target_idx,
            "replacement": replacement,
        })

        with torch.no_grad():
            outputs_repl = self.model(**inputs)
        logits_repl = outputs_repl.logits

        # Compute JS divergence
        p = F.softmax(logits[0, -1].float(), dim=-1)
        q = F.softmax(logits_repl[0, -1].float(), dim=-1)
        js = js_divergence(p, q)
        log.info(f"  JS divergence (center replacement): {js:.6f}")
        results["probe_js_divergence"] = js

        self._visual_replace_cfg["active"] = False
        self._remove_visual_hook()

        log.info("\n  P0-a PASSED ✓" if js > 1e-6 else "\n  P0-a WARNING: JS ≈ 0")
        log.info("=" * 60)
        return results

    # ---- Core CED computation for a single sample ----

    def compute_sample_ced(
        self, image: Image.Image, question: str, target_bbox: list,
        image_size: tuple, layer_indices: list,
    ) -> dict:
        """Compute CED and all metric variants for one sample.

        Returns dict with keys:
          js_logits, kl_logits, cos_logits, ent_orig, ent_repl,
          js_layer_{i}, kl_layer_{i}, cos_layer_{i} for each layer.
        """
        # Prepare inputs
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image],
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Get visual features for replacement computation
        pv = inputs["pixel_values"].to(dtype=self.dtype)
        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is not None:
            grid_thw = grid_thw.to(self.device)

        with torch.no_grad():
            vis_features = self.model.visual(pv, grid_thw=grid_thw)

        n_vis = vis_features.shape[0]

        # Map bbox to merged token indices
        W_res, H_res = get_resized_image_size(image)
        target_indices = bbox_to_merged_token_indices(
            target_bbox, image_size, (W_res, H_res)
        )
        # Clamp indices
        target_indices = [i for i in target_indices if 0 <= i < n_vis]
        if not target_indices:
            # Fallback: use center tokens
            center = n_vis // 2
            target_indices = [center]

        surround_indices = [i for i in range(n_vis) if i not in set(target_indices)]
        if not surround_indices:
            surround_indices = list(range(n_vis))
        replacement = vis_features[surround_indices].mean(dim=0)

        # Setup hooks
        self._setup_visual_hook()
        int_layers = [int(l) for l in layer_indices if l != "logits"]
        self.register_layer_hooks(int_layers)

        result = {}

        # --- Forward pass 1: Original ---
        self._visual_replace_cfg["active"] = False
        self._capture_mode = "orig"
        self._layer_states_orig.clear()
        with torch.no_grad():
            outputs_orig = self.model(**inputs)
        logits_orig = outputs_orig.logits[0, -1].float()  # (vocab,)

        # --- Forward pass 2: Replaced ---
        self._visual_replace_cfg.update({
            "active": True,
            "target_indices": target_indices,
            "replacement": replacement,
        })
        self._capture_mode = "repl"
        self._layer_states_repl.clear()
        with torch.no_grad():
            outputs_repl = self.model(**inputs)
        logits_repl = outputs_repl.logits[0, -1].float()

        # Cleanup hooks
        self._visual_replace_cfg["active"] = False
        self._remove_visual_hook()

        # --- Compute metrics at logits level ---
        p = F.softmax(logits_orig, dim=-1)
        q = F.softmax(logits_repl, dim=-1)

        result["js_logits"] = js_divergence(p, q)
        result["kl_logits"] = kl_divergence(p, q)
        result["cos_logits"] = cosine_distance(logits_orig, logits_repl)
        result["ent_orig"] = entropy(p)
        result["ent_repl"] = entropy(q)
        result["n_target_tokens"] = len(target_indices)
        result["n_total_vis_tokens"] = n_vis

        # --- Compute metrics at intermediate layers ---
        lm_head = self.model.lm_head
        for layer_idx in int_layers:
            if (layer_idx in self._layer_states_orig and
                    layer_idx in self._layer_states_repl):
                h_orig = self._layer_states_orig[layer_idx].float()
                h_repl = self._layer_states_repl[layer_idx].float()

                # Project through LM head to get pseudo-distributions
                with torch.no_grad():
                    pl_orig = F.softmax(lm_head(h_orig).squeeze(0), dim=-1)
                    pl_repl = F.softmax(lm_head(h_repl).squeeze(0), dim=-1)

                result[f"js_layer_{layer_idx}"] = js_divergence(pl_orig, pl_repl)
                result[f"kl_layer_{layer_idx}"] = kl_divergence(pl_orig, pl_repl)
                result[f"cos_layer_{layer_idx}"] = cosine_distance(h_orig.squeeze(),
                                                                    h_repl.squeeze())

        self.remove_hooks()
        return result

    # ---- Generate model answer ----

    def generate_answer(self, image: Image.Image, question: str,
                        max_new_tokens: int = 32) -> str:
        """Generate a text answer from the model."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=[image],
                                return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0,
            )

        # Decode only new tokens
        prompt_len = inputs["input_ids"].shape[1]
        answer = self.processor.tokenizer.decode(
            gen_ids[0][prompt_len:], skip_special_tokens=True
        ).strip()
        return answer

    # ---- Behavior classification ----

    @staticmethod
    def classify_behavior(answer: str, gt: str, task: str) -> str:
        """Classify model behavior into four groups.

        For yes/no tasks:
          correct_positive: model says yes, gt is yes
          hallucination:    model says yes, gt is no
          correct_negative: model says no, gt is no
          miss:             model says no, gt is yes

        For other tasks:
          correct / incorrect
        """
        answer_lower = answer.lower().strip()

        if task in ("existence", "spatial"):
            model_yes = any(w in answer_lower for w in ["yes", "there is", "correct"])
            model_no = any(w in answer_lower for w in ["no", "there isn't",
                                                        "there is no", "not"])
            if not model_yes and not model_no:
                model_yes = "yes" in answer_lower

            gt_yes = gt.lower().strip() == "yes"

            if model_yes and gt_yes:
                return "correct_positive"
            elif model_yes and not gt_yes:
                return "hallucination"
            elif not model_yes and not gt_yes:
                return "correct_negative"
            else:
                return "miss"
        else:
            # For attribute/counting: simple match
            if gt.lower().strip() in answer_lower:
                return "correct"
            return "incorrect"


# ============================================================================
# 5. P0-b: Full validation suite
# ============================================================================

def run_validation(exp: CEDExperiment, samples: dict, layer_indices: list,
                   lambda_e_values: list, output_dir: str):
    """Run all P0-b experiments and save results."""
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    # Flatten all samples
    flat_samples = []
    for task, task_samples in samples.items():
        flat_samples.extend(task_samples)
    log.info(f"Total samples to process: {len(flat_samples)}")

    # ---- Phase 1: Generate answers + classify behavior ----
    log.info("Phase 1: Generating answers and classifying behavior...")
    for s in tqdm(flat_samples, desc="Generating answers"):
        image = Image.open(s["img_path"]).convert("RGB")
        s["model_answer"] = exp.generate_answer(image, s["question"])
        s["behavior"] = exp.classify_behavior(
            s["model_answer"], s["answer_gt"], s["task"]
        )

    # Print behavior distribution
    behavior_counts = defaultdict(int)
    for s in flat_samples:
        behavior_counts[s["behavior"]] += 1
    log.info(f"Behavior distribution: {dict(behavior_counts)}")

    # ---- Phase 2: Compute CED for samples with target_bbox ----
    log.info("Phase 2: Computing CED metrics...")
    ced_samples = [s for s in flat_samples if s["target_bbox"] is not None]
    log.info(f"Samples with target bbox: {len(ced_samples)}")

    for s in tqdm(ced_samples, desc="Computing CED"):
        image = Image.open(s["img_path"]).convert("RGB")
        try:
            metrics = exp.compute_sample_ced(
                image, s["question"], s["target_bbox"],
                s["image_size"], layer_indices,
            )
            s["metrics"] = metrics
        except Exception as e:
            log.warning(f"Error computing CED for img {s['img_id']}: {e}")
            s["metrics"] = None

    # Filter out failed samples
    valid_samples = [s for s in ced_samples if s["metrics"] is not None]
    log.info(f"Valid CED samples: {len(valid_samples)}")

    # ---- Compute CED for all lambda_e values ----
    for s in valid_samples:
        m = s["metrics"]
        s["ced_values"] = {}
        for lam in lambda_e_values:
            s["ced_values"][str(lam)] = compute_ced(
                m["js_logits"], m["ent_orig"], m["ent_repl"], lam
            )

    # ---- Analysis ----
    results_summary = {}

    # A. Behavior grouping analysis (existence task, yes/no only)
    log.info("\n" + "=" * 60)
    log.info("Analysis A: Behavior Grouping (Existence Task)")
    log.info("=" * 60)
    existence_valid = [s for s in valid_samples if s["task"] == "existence"]
    results_summary["behavior_grouping"] = analyze_behavior_groups(
        existence_valid, lambda_e_values, output_dir
    )

    # B. Formula ablation
    log.info("\n" + "=" * 60)
    log.info("Analysis B: Formula Ablation")
    log.info("=" * 60)
    results_summary["formula_ablation"] = analyze_formula_ablation(
        existence_valid, layer_indices, lambda_e_values, output_dir
    )

    # C. Cross-layer analysis
    log.info("\n" + "=" * 60)
    log.info("Analysis C: Cross-Layer Analysis")
    log.info("=" * 60)
    results_summary["cross_layer"] = analyze_cross_layer(
        existence_valid, layer_indices, output_dir
    )

    # D. Cross-task consistency
    log.info("\n" + "=" * 60)
    log.info("Analysis D: Cross-Task Consistency")
    log.info("=" * 60)
    results_summary["cross_task"] = analyze_cross_task(
        valid_samples, lambda_e_values, output_dir
    )

    # ---- Save all results ----
    # Serialize for JSON
    serializable = []
    for s in flat_samples:
        entry = {k: v for k, v in s.items()
                 if k not in ("metrics", "ced_values")}
        if "metrics" in s and s.get("metrics"):
            entry["metrics"] = s["metrics"]
        if "ced_values" in s:
            entry["ced_values"] = s["ced_values"]
        serializable.append(entry)

    json_path = os.path.join(output_dir, "ced_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "samples": serializable,
            "summary": _make_serializable(results_summary),
        }, f, indent=2, default=str)
    log.info(f"\nAll results saved to {json_path}")

    # Print final summary
    print_final_summary(results_summary)

    return results_summary


def _make_serializable(obj):
    """Convert numpy/torch types to Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (torch.Tensor,)):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return obj


# ============================================================================
# 6. Analysis functions
# ============================================================================

def analyze_behavior_groups(samples: list, lambda_e_values: list,
                            output_dir: str) -> dict:
    """Analyze CED distribution across behavior groups.
    
    Core hypothesis: correct_positive should have HIGHER CED than hallucination.
    """
    groups = defaultdict(list)
    for s in samples:
        if s.get("metrics"):
            groups[s["behavior"]].append(s)

    results = {}
    for behavior, group_samples in groups.items():
        js_vals = [s["metrics"]["js_logits"] for s in group_samples]
        results[behavior] = {
            "count": len(group_samples),
            "js_mean": float(np.mean(js_vals)),
            "js_std": float(np.std(js_vals)),
            "js_median": float(np.median(js_vals)),
        }
        log.info(f"  {behavior:20s}: n={len(group_samples):4d}, "
                 f"JS={np.mean(js_vals):.6f} ± {np.std(js_vals):.6f}")

    # AUC-ROC: correct_positive vs hallucination
    cp = [s["metrics"]["js_logits"] for s in groups.get("correct_positive", [])]
    hal = [s["metrics"]["js_logits"] for s in groups.get("hallucination", [])]

    if len(cp) >= 5 and len(hal) >= 5:
        labels = [1] * len(cp) + [0] * len(hal)
        scores = cp + hal
        auc = roc_auc_score(labels, scores)
        results["auc_cp_vs_hal_js"] = float(auc)
        log.info(f"\n  AUC (correct_positive vs hallucination, JS): {auc:.4f}")

        # Mann-Whitney U test
        stat, pval = mannwhitneyu(cp, hal, alternative="greater")
        results["mannwhitney_pval"] = float(pval)
        log.info(f"  Mann-Whitney U p-value: {pval:.2e}")

        # AUC for different lambda_e
        for lam in lambda_e_values:
            cp_ced = [s["ced_values"][str(lam)] for s in groups.get("correct_positive", [])]
            hal_ced = [s["ced_values"][str(lam)] for s in groups.get("hallucination", [])]
            if cp_ced and hal_ced:
                labels_ced = [1] * len(cp_ced) + [0] * len(hal_ced)
                auc_ced = roc_auc_score(labels_ced, cp_ced + hal_ced)
                results[f"auc_ced_lambda_{lam}"] = float(auc_ced)
                log.info(f"  AUC (CED, λ_e={lam}): {auc_ced:.4f}")

        # Plot distributions
        _plot_behavior_distributions(groups, output_dir)
        _plot_roc_curve(cp, hal, output_dir)
    else:
        log.warning(f"  Not enough samples for AUC: "
                    f"correct_positive={len(cp)}, hallucination={len(hal)}")
        results["auc_cp_vs_hal_js"] = None

    return results


def _plot_behavior_distributions(groups: dict, output_dir: str):
    """Plot JS divergence distribution for each behavior group."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    colors = {
        "correct_positive": "#2ecc71",
        "hallucination": "#e74c3c",
        "correct_negative": "#3498db",
        "miss": "#f39c12",
    }
    for behavior, group_samples in groups.items():
        if not group_samples:
            continue
        vals = [s["metrics"]["js_logits"] for s in group_samples]
        ax.hist(vals, bins=30, alpha=0.5, label=f"{behavior} (n={len(vals)})",
                color=colors.get(behavior, "#95a5a6"), density=True)

    ax.set_xlabel("JS Divergence (logits)")
    ax.set_ylabel("Density")
    ax.set_title("CED Signal Distribution by Behavior Group")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_behavior_distribution.png"), dpi=150)
    plt.close(fig)
    log.info(f"  Saved: fig_behavior_distribution.png")


def _plot_roc_curve(cp_scores: list, hal_scores: list, output_dir: str):
    """Plot ROC curve for correct_positive vs hallucination."""
    labels = [1] * len(cp_scores) + [0] * len(hal_scores)
    scores = cp_scores + hal_scores
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"JS Divergence (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC: Correct Positive vs Hallucination")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_roc_curve.png"), dpi=150)
    plt.close(fig)
    log.info(f"  Saved: fig_roc_curve.png")


def analyze_formula_ablation(samples: list, layer_indices: list,
                             lambda_e_values: list, output_dir: str) -> dict:
    """Compare different divergence metrics and CED variants.

    Metrics: raw JS, CED (various λ_e), KL, cosine, entropy-only.
    """
    # Separate positive-labeled samples for AUC
    cp = [s for s in samples if s["behavior"] == "correct_positive" and s.get("metrics")]
    hal = [s for s in samples if s["behavior"] == "hallucination" and s.get("metrics")]

    if len(cp) < 5 or len(hal) < 5:
        log.warning("Not enough samples for formula ablation AUC")
        return {"error": "insufficient_samples"}

    results = {}
    metric_aucs = {}

    # 1. Raw JS
    labels = [1] * len(cp) + [0] * len(hal)
    js_scores = [s["metrics"]["js_logits"] for s in cp] + \
                [s["metrics"]["js_logits"] for s in hal]
    metric_aucs["JS"] = roc_auc_score(labels, js_scores)

    # 2. KL
    kl_scores = [s["metrics"]["kl_logits"] for s in cp] + \
                [s["metrics"]["kl_logits"] for s in hal]
    metric_aucs["KL"] = roc_auc_score(labels, kl_scores)

    # 3. Cosine
    cos_scores = [s["metrics"]["cos_logits"] for s in cp] + \
                 [s["metrics"]["cos_logits"] for s in hal]
    metric_aucs["Cosine"] = roc_auc_score(labels, cos_scores)

    # 4. Entropy-only (ent_repl - ent_orig)
    ent_scores = [s["metrics"]["ent_repl"] - s["metrics"]["ent_orig"] for s in cp] + \
                 [s["metrics"]["ent_repl"] - s["metrics"]["ent_orig"] for s in hal]
    metric_aucs["Entropy_diff"] = roc_auc_score(labels, ent_scores)

    # 5. CED with different λ_e
    for lam in lambda_e_values:
        ced_scores = [s["ced_values"][str(lam)] for s in cp] + \
                     [s["ced_values"][str(lam)] for s in hal]
        metric_aucs[f"CED_λ={lam}"] = roc_auc_score(labels, ced_scores)

    # Print and store
    log.info("  Formula Ablation AUC-ROC:")
    for metric, auc in sorted(metric_aucs.items(), key=lambda x: -x[1]):
        log.info(f"    {metric:20s}: {auc:.4f}")
        results[metric] = float(auc)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    names = list(metric_aucs.keys())
    aucs = [metric_aucs[n] for n in names]
    colors_bar = ["#e74c3c" if "CED" in n else "#3498db" for n in names]
    bars = ax.barh(names, aucs, color=colors_bar, alpha=0.8)
    ax.axvline(0.85, color="green", linestyle="--", alpha=0.7, label="Target AUC=0.85")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Formula Ablation: AUC for Different Metrics")
    ax.legend()
    ax.set_xlim(0.3, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_formula_ablation.png"), dpi=150)
    plt.close(fig)
    log.info(f"  Saved: fig_formula_ablation.png")

    return results


def analyze_cross_layer(samples: list, layer_indices: list,
                        output_dir: str) -> dict:
    """Compare AUC at different transformer layers."""
    cp = [s for s in samples if s["behavior"] == "correct_positive" and s.get("metrics")]
    hal = [s for s in samples if s["behavior"] == "hallucination" and s.get("metrics")]

    if len(cp) < 5 or len(hal) < 5:
        log.warning("Not enough samples for cross-layer analysis")
        return {"error": "insufficient_samples"}

    results = {}
    labels = [1] * len(cp) + [0] * len(hal)

    layer_aucs = {}

    # Logits layer
    js_logits = [s["metrics"]["js_logits"] for s in cp] + \
                [s["metrics"]["js_logits"] for s in hal]
    layer_aucs["logits"] = roc_auc_score(labels, js_logits)

    # Intermediate layers
    for l in layer_indices:
        if l == "logits":
            continue
        key = f"js_layer_{l}"
        vals_cp = [s["metrics"].get(key) for s in cp if s["metrics"].get(key) is not None]
        vals_hal = [s["metrics"].get(key) for s in hal if s["metrics"].get(key) is not None]
        if len(vals_cp) >= 5 and len(vals_hal) >= 5:
            labels_l = [1] * len(vals_cp) + [0] * len(vals_hal)
            layer_aucs[f"layer_{l}"] = roc_auc_score(labels_l, vals_cp + vals_hal)

    log.info("  Cross-Layer AUC-ROC (JS divergence):")
    for layer, auc in sorted(layer_aucs.items()):
        log.info(f"    {layer:15s}: {auc:.4f}")
        results[layer] = float(auc)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    layer_names = sorted(layer_aucs.keys(),
                         key=lambda x: int(x.split("_")[-1]) if x != "logits" else 999)
    aucs = [layer_aucs[n] for n in layer_names]
    ax.plot(range(len(layer_names)), aucs, "bo-", markersize=8, linewidth=2)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=45)
    ax.axhline(0.85, color="green", linestyle="--", alpha=0.7, label="Target=0.85")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Cross-Layer Analysis: JS Divergence AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_cross_layer.png"), dpi=150)
    plt.close(fig)
    log.info(f"  Saved: fig_cross_layer.png")

    return results


def analyze_cross_task(samples: list, lambda_e_values: list,
                       output_dir: str) -> dict:
    """Check if CED signal is consistent across task types."""
    results = {}
    tasks = ["existence", "spatial", "attribute", "counting"]

    for task in tasks:
        task_samples = [s for s in samples if s["task"] == task and s.get("metrics")]
        if not task_samples:
            continue

        js_vals = [s["metrics"]["js_logits"] for s in task_samples]

        # For existence/spatial: compute AUC if we have both groups
        if task in ("existence", "spatial"):
            cp = [s for s in task_samples if s["behavior"] == "correct_positive"]
            hal = [s for s in task_samples if s["behavior"] == "hallucination"]
            if len(cp) >= 3 and len(hal) >= 3:
                labels = [1] * len(cp) + [0] * len(hal)
                scores = ([s["metrics"]["js_logits"] for s in cp] +
                          [s["metrics"]["js_logits"] for s in hal])
                auc = roc_auc_score(labels, scores)
                results[task] = {
                    "n_samples": len(task_samples),
                    "js_mean": float(np.mean(js_vals)),
                    "js_std": float(np.std(js_vals)),
                    "auc": float(auc),
                    "n_cp": len(cp),
                    "n_hal": len(hal),
                }
                log.info(f"  {task:12s}: n={len(task_samples)}, "
                         f"JS={np.mean(js_vals):.6f}, AUC={auc:.4f}")
            else:
                results[task] = {
                    "n_samples": len(task_samples),
                    "js_mean": float(np.mean(js_vals)),
                    "js_std": float(np.std(js_vals)),
                    "auc": None,
                }
                log.info(f"  {task:12s}: n={len(task_samples)}, "
                         f"JS={np.mean(js_vals):.6f}, AUC=N/A (insufficient groups)")
        else:
            # For attribute/counting: report stats only
            correct = [s for s in task_samples if s["behavior"] == "correct"]
            incorrect = [s for s in task_samples if s["behavior"] == "incorrect"]
            if len(correct) >= 3 and len(incorrect) >= 3:
                labels = [1] * len(correct) + [0] * len(incorrect)
                scores = ([s["metrics"]["js_logits"] for s in correct] +
                          [s["metrics"]["js_logits"] for s in incorrect])
                auc = roc_auc_score(labels, scores)
            else:
                auc = None
            results[task] = {
                "n_samples": len(task_samples),
                "js_mean": float(np.mean(js_vals)),
                "js_std": float(np.std(js_vals)),
                "auc": float(auc) if auc else None,
            }
            log.info(f"  {task:12s}: n={len(task_samples)}, "
                     f"JS={np.mean(js_vals):.6f}, "
                     f"AUC={'N/A' if auc is None else f'{auc:.4f}'}")

    # Plot cross-task comparison
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 5),
                             sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        task_samples = [s for s in samples if s["task"] == task and s.get("metrics")]
        if not task_samples:
            continue
        js_vals = [s["metrics"]["js_logits"] for s in task_samples]
        ax.hist(js_vals, bins=20, alpha=0.7, color="#3498db")
        ax.set_title(f"{task}\n(n={len(task_samples)})")
        ax.set_xlabel("JS Divergence")
        if task in results and results[task].get("auc"):
            ax.text(0.95, 0.95, f"AUC={results[task]['auc']:.3f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat"))

    axes[0].set_ylabel("Count")
    fig.suptitle("Cross-Task JS Divergence Distribution", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig_cross_task.png"), dpi=150)
    plt.close(fig)
    log.info(f"  Saved: fig_cross_task.png")

    return results


# ============================================================================
# 7. Summary printing
# ============================================================================

def print_final_summary(results: dict):
    """Print a clean final summary of all P0-b experiments."""
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY: CED Phase 0-b Results")
    log.info("=" * 70)

    # Behavior grouping
    bg = results.get("behavior_grouping", {})
    auc_main = bg.get("auc_cp_vs_hal_js")
    pval = bg.get("mannwhitney_pval")
    log.info(f"\n[Behavior Grouping]")
    for beh in ["correct_positive", "hallucination", "correct_negative", "miss"]:
        if beh in bg:
            info = bg[beh]
            log.info(f"  {beh:20s}: n={info['count']}, "
                     f"JS={info['js_mean']:.6f} ± {info['js_std']:.6f}")
    if auc_main is not None:
        status = "✓ PASS" if auc_main >= 0.85 else \
                 "~ MARGINAL" if auc_main >= 0.75 else "✗ FAIL"
        log.info(f"\n  AUC (cp vs hal): {auc_main:.4f}  [{status}]")
        log.info(f"  p-value:         {pval:.2e}")

    # Best CED config
    best_ced_key = None
    best_ced_auc = 0
    for k, v in bg.items():
        if k.startswith("auc_ced_lambda_") and v is not None and v > best_ced_auc:
            best_ced_auc = v
            best_ced_key = k
    if best_ced_key:
        log.info(f"  Best CED config: {best_ced_key} → AUC={best_ced_auc:.4f}")

    # Formula ablation
    fa = results.get("formula_ablation", {})
    if fa and "error" not in fa:
        log.info(f"\n[Formula Ablation] (sorted by AUC)")
        for metric, auc in sorted(fa.items(), key=lambda x: -x[1]):
            marker = " ← best" if auc == max(fa.values()) else ""
            log.info(f"  {metric:20s}: {auc:.4f}{marker}")

    # Cross-layer
    cl = results.get("cross_layer", {})
    if cl and "error" not in cl:
        log.info(f"\n[Cross-Layer] (JS divergence AUC)")
        best_layer = max(cl.items(), key=lambda x: x[1])
        for layer, auc in sorted(cl.items()):
            marker = " ← best" if auc == best_layer[1] else ""
            log.info(f"  {layer:15s}: {auc:.4f}{marker}")

    # Cross-task
    ct = results.get("cross_task", {})
    if ct:
        log.info(f"\n[Cross-Task Consistency]")
        for task, info in ct.items():
            auc_str = f"AUC={info['auc']:.4f}" if info.get("auc") else "AUC=N/A"
            log.info(f"  {task:12s}: n={info['n_samples']}, "
                     f"JS={info['js_mean']:.6f}, {auc_str}")

    # Overall verdict
    log.info(f"\n{'=' * 70}")
    if auc_main is not None and auc_main >= 0.85:
        log.info("VERDICT: P0 PASSED ✓  — CED signal has sufficient discriminative "
                 "power. Proceed to P1.")
    elif auc_main is not None and auc_main >= 0.75:
        log.info("VERDICT: P0 MARGINAL ~  — AUC between 0.75-0.85. Consider "
                 "tuning replacement strategy or trying Plan B/C/D.")
    else:
        log.info("VERDICT: P0 NEEDS WORK ✗  — AUC < 0.75. Try alternative "
                 "replacement methods: VCD noise / MaskCD / PROJECTAWAY.")
    log.info("=" * 70)


# ============================================================================
# 8. Main entry point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CED Phase 0: Counterfactual Evidence Divergence Validation"
    )
    parser.add_argument("--mode", choices=["probe", "validate", "both"],
                        default="both",
                        help="probe=P0-a only, validate=P0-b, both=P0-a+P0-b")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--coco_img_dir", type=str, required=True,
                        help="Path to COCO val2017 images")
    parser.add_argument("--coco_ann_file", type=str, required=True,
                        help="Path to instances_val2017.json")
    parser.add_argument("--output_dir", type=str, default="./results_p0",
                        help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Total number of samples (split across 4 tasks)")
    parser.add_argument("--layers", nargs="+", default=["logits", "12", "16", "20", "24", "27"],
                        help="Layers to analyze (include 'logits')")
    parser.add_argument("--lambda_e_values", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5],
                        help="Lambda_e values for CED formula ablation")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log.info("CED Phase 0 Experiment")
    log.info(f"  Model:    {args.model_path}")
    log.info(f"  COCO:     {args.coco_img_dir}")
    log.info(f"  Output:   {args.output_dir}")
    log.info(f"  Mode:     {args.mode}")
    log.info(f"  Samples:  {args.num_samples}")
    log.info(f"  Layers:   {args.layers}")
    log.info(f"  Lambda_e: {args.lambda_e_values}")

    # Initialize experiment
    exp = CEDExperiment(args.model_path, device=args.device)

    # ---- P0-a: Architecture Probing ----
    if args.mode in ("probe", "both"):
        # Find a sample image
        sample_img = None
        for f in os.listdir(args.coco_img_dir):
            if f.endswith(".jpg"):
                sample_img = os.path.join(args.coco_img_dir, f)
                break
        if sample_img is None:
            log.error("No .jpg images found in COCO dir!")
            sys.exit(1)

        probe_results = exp.probe_architecture(sample_img)

        # Save probe results
        probe_path = os.path.join(args.output_dir, "p0a_probe_results.json")
        with open(probe_path, "w") as f:
            json.dump(probe_results, f, indent=2, default=str)
        log.info(f"P0-a results saved to {probe_path}")

        if args.mode == "probe":
            return

    # ---- P0-b: CED Validation ----
    if args.mode in ("validate", "both"):
        log.info("\nLoading COCO annotations...")
        coco = COCO(args.coco_ann_file)

        log.info("Preparing samples...")
        samples = prepare_samples(coco, args.coco_img_dir,
                                  num_samples=args.num_samples, seed=args.seed)

        log.info("Running CED validation suite...")
        t0 = time.time()
        results = run_validation(
            exp, samples, args.layers, args.lambda_e_values, args.output_dir
        )
        elapsed = time.time() - t0
        log.info(f"\nTotal validation time: {elapsed / 60:.1f} minutes")

    log.info("\nDone! Check output directory for results and figures.")


if __name__ == "__main__":
    main()
