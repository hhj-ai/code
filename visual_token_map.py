"""bbox → visual token index 映射（含 Qwen3-VL 2×2 merger）。"""

import math
from typing import List, Tuple


def bbox_to_token_indices(bbox, img_w, img_h, grid_h, grid_w) -> List[int]:
    """bbox [x,y,w,h] → merged visual token 线性索引。"""
    x, y, w, h = bbox
    c0 = max(0, min(int(math.floor(x / img_w * grid_w)), grid_w - 1))
    c1 = max(c0 + 1, min(int(math.ceil((x + w) / img_w * grid_w)), grid_w))
    r0 = max(0, min(int(math.floor(y / img_h * grid_h)), grid_h - 1))
    r1 = max(r0 + 1, min(int(math.ceil((y + h) / img_h * grid_h)), grid_h))
    return [r * grid_w + c for r in range(r0, r1) for c in range(c0, c1)]


def surrounding_indices(target, grid_h, grid_w, ring=2) -> List[int]:
    """目标区域周围环形邻域索引。"""
    tgt = set(target)
    sur = set()
    for idx in target:
        r, c = idx // grid_w, idx % grid_w
        for dr in range(-ring, ring + 1):
            for dc in range(-ring, ring + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    nid = nr * grid_w + nc
                    if nid not in tgt:
                        sur.add(nid)
    return sorted(sur)


def find_visual_range(input_ids, image_token_id: int) -> Tuple[int, int]:
    """返回 input_ids 中 visual token 的 [start, end)。"""
    ids = input_ids.squeeze().tolist()
    pos = [i for i, t in enumerate(ids) if t == image_token_id]
    assert pos, f"image_token_id={image_token_id} not found in input_ids"
    return pos[0], pos[-1] + 1


def to_absolute(input_ids, rel_indices, image_token_id) -> List[int]:
    """相对索引 → input_ids 绝对位置。"""
    vs, ve = find_visual_range(input_ids, image_token_id)
    n = ve - vs
    return [vs + i for i in rel_indices if i < n]
