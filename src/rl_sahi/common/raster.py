from __future__ import annotations

import numpy as np

from rl_sahi.common.box_types import EPS, as_boxes


def rasterize_boxes(
    boxes: np.ndarray,
    image_shape: tuple[int, int],
    grid_size: int,
    values: np.ndarray | None = None,
    fill_mode: str = "max",
) -> np.ndarray:
    boxes = as_boxes(boxes)
    if len(boxes) == 0:
        return np.zeros((grid_size, grid_size), dtype=np.float32)
        
    h, w = image_shape
    if values is None:
        values = np.ones((len(boxes),), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)

    x1 = np.floor((boxes[:, 0] / max(w, EPS)) * grid_size).astype(np.int32)
    y1 = np.floor((boxes[:, 1] / max(h, EPS)) * grid_size).astype(np.int32)
    x2 = np.ceil((boxes[:, 2] / max(w, EPS)) * grid_size).astype(np.int32)
    y2 = np.ceil((boxes[:, 3] / max(h, EPS)) * grid_size).astype(np.int32)
    
    x1 = np.clip(x1, 0, grid_size - 1)
    y1 = np.clip(y1, 0, grid_size - 1)
    x2 = np.clip(x2, x1 + 1, grid_size)
    y2 = np.clip(y2, y1 + 1, grid_size)

    Y, X = np.ogrid[0:grid_size, 0:grid_size]
    mask = (X >= x1[:, None, None]) & (X < x2[:, None, None]) & \
           (Y >= y1[:, None, None]) & (Y < y2[:, None, None])
           
    weighted_mask = mask * values[:, None, None]
    
    if fill_mode == "add":
        grid = weighted_mask.sum(axis=0)
    else:
        grid = weighted_mask.max(axis=0)
        
    return np.clip(grid, 0.0, 1.0).astype(np.float32)
