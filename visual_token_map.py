"""visual_token_map.py — bbox → visual token index 映射。

Qwen3-VL 的图像处理流程：
1. 图像被 resize 到特定分辨率（由 processor 决定）
2. ViT 编码器将图像分成 patch（patch_size = 14）
3. 2×2 Merger 将 4 个相邻 patch token 合并为 1 个 merged token
4. Merged token 和文本 token 拼接后送入 LLM

因此 bbox → token index 的映射需要：
a) 知道图像实际被 resize 到了多大
b) 转换到 patch 坐标
c) 考虑 2×2 merger 的降采样
d) 定位在 LLM 输入序列中的位置
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional, Dict, Any

import torch
import numpy as np


def get_visual_token_grid(
    image_grid_thw: torch.Tensor,
) -> Tuple[int, int, int]:
    """从 processor 输出的 image_grid_thw 获取网格尺寸。
    
    image_grid_thw: shape [n_images, 3]，每行是 (t, h, w)
    t: temporal（静态图=1）
    h, w: merger 之后的 grid 高宽
    
    返回 (temporal, grid_h, grid_w)
    """
    if image_grid_thw.dim() == 1:
        t, h, w = image_grid_thw.tolist()
    else:
        t, h, w = image_grid_thw[0].tolist()
    return int(t), int(h), int(w)


def bbox_to_merged_token_indices(
    bbox: List[float],
    image_width: int,
    image_height: int,
    grid_h: int,
    grid_w: int,
    temporal: int = 1,
    merge_size: int = 2,
    patch_size: int = 14,
) -> List[int]:
    """将原始图像上的 bbox [x, y, w, h] 映射到 merged visual token indices。

    流程：
    1. bbox 坐标 → 归一化坐标 [0, 1]
    2. 归一化坐标 → merged grid 坐标
    3. 取覆盖区域内所有 merged token 的线性 index

    Args:
        bbox: [x, y, w, h] 原始图像坐标
        image_width, image_height: 原始图像尺寸
        grid_h, grid_w: merger 后的网格尺寸（来自 image_grid_thw）
        temporal: 时序维度（静态图=1）
        merge_size: merger 的合并因子（Qwen3-VL = 2）
        patch_size: ViT patch 大小（Qwen3-VL = 14）

    Returns:
        merged token 的线性索引列表（在 visual token 序列中的位置）
    """
    x, y, w, h = bbox

    # 归一化到 [0, 1]
    x_min_norm = max(0.0, x / image_width)
    y_min_norm = max(0.0, y / image_height)
    x_max_norm = min(1.0, (x + w) / image_width)
    y_max_norm = min(1.0, (y + h) / image_height)

    # 映射到 merged grid 坐标
    col_min = int(math.floor(x_min_norm * grid_w))
    col_max = int(math.ceil(x_max_norm * grid_w))
    row_min = int(math.floor(y_min_norm * grid_h))
    row_max = int(math.ceil(y_max_norm * grid_h))

    # clamp
    col_min = max(0, min(col_min, grid_w - 1))
    col_max = max(col_min + 1, min(col_max, grid_w))
    row_min = max(0, min(row_min, grid_h - 1))
    row_max = max(row_min + 1, min(row_max, grid_h))

    # 收集线性索引（row-major order）
    indices = []
    for r in range(row_min, row_max):
        for c in range(col_min, col_max):
            idx = r * grid_w + c
            indices.append(idx)

    return indices


def get_surrounding_token_indices(
    target_indices: List[int],
    grid_h: int,
    grid_w: int,
    ring_width: int = 1,
) -> List[int]:
    """获取目标 token 区域周围的 token indices（环形邻域）。
    
    用于计算"周围 token 均值"作为替换值。
    """
    target_set = set(target_indices)
    surrounding = set()

    for idx in target_indices:
        r = idx // grid_w
        c = idx % grid_w
        for dr in range(-ring_width, ring_width + 1):
            for dc in range(-ring_width, ring_width + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    new_idx = nr * grid_w + nc
                    if new_idx not in target_set:
                        surrounding.add(new_idx)

    return sorted(surrounding)


def find_visual_token_range_in_input_ids(
    input_ids: torch.Tensor,
    image_token_id: int,
    vision_start_token_id: Optional[int] = None,
    vision_end_token_id: Optional[int] = None,
) -> Tuple[int, int]:
    """在 input_ids 序列中找到 visual token 的起止位置。

    Qwen3-VL 的 input_ids 中，visual token 用特殊的 image_token_id 填充。
    返回 (start_pos, end_pos)，即 input_ids 中 visual token 的范围。
    """
    ids = input_ids.squeeze().tolist()

    # 方法1：找连续的 image_token_id 块
    positions = [i for i, tok in enumerate(ids) if tok == image_token_id]
    if positions:
        return positions[0], positions[-1] + 1

    # 方法2：找 vision_start / vision_end token
    if vision_start_token_id is not None and vision_end_token_id is not None:
        start = None
        for i, tok in enumerate(ids):
            if tok == vision_start_token_id:
                start = i + 1
            if tok == vision_end_token_id and start is not None:
                return start, i

    raise ValueError("Cannot find visual token range in input_ids")


def visual_token_absolute_positions(
    input_ids: torch.Tensor,
    target_token_indices: List[int],
    image_token_id: int,
    vision_start_token_id: Optional[int] = None,
    vision_end_token_id: Optional[int] = None,
) -> List[int]:
    """将 visual token 的相对索引转换为 input_ids 中的绝对位置。

    Args:
        target_token_indices: 在 visual token 序列中的相对位置
        其他参数用于定位 visual token 在 input_ids 中的范围

    Returns:
        input_ids 中的绝对位置列表
    """
    vis_start, vis_end = find_visual_token_range_in_input_ids(
        input_ids, image_token_id, vision_start_token_id, vision_end_token_id
    )
    n_visual = vis_end - vis_start

    abs_positions = []
    for idx in target_token_indices:
        if idx < n_visual:
            abs_positions.append(vis_start + idx)
        # 超出范围的忽略（bbox可能超出图像边界）

    return abs_positions
