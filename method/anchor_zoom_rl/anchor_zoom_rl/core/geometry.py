from __future__ import annotations

import numpy as np

from .types import Anchor


def box_area(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    wh = np.maximum(boxes[:, 2:4] - boxes[:, 0:2], 0.0)
    return wh[:, 0] * wh[:, 1]


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(bottom_right - top_left, 0.0)
    intersection = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - intersection
    return intersection / np.maximum(union, 1e-7)


def ios_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Intersection over the smaller box area."""
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(bottom_right - top_left, 0.0)
    intersection = wh[..., 0] * wh[..., 1]
    smaller = np.minimum(box_area(a)[:, None], box_area(b)[None, :])
    return intersection / np.maximum(smaller, 1e-7)


def clip_box(box: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    result = np.asarray(box, dtype=np.float32).copy()
    result[[0, 2]] = np.clip(result[[0, 2]], 0.0, float(width))
    result[[1, 3]] = np.clip(result[[1, 3]], 0.0, float(height))
    return result


def anchor_to_roi(
    anchor: Anchor,
    zoom: float,
    image_shape: tuple[int, int],
    base_slice_fraction: float,
    min_slice_fraction: float,
    max_slice_fraction: float,
    context_factor: float,
) -> np.ndarray:
    height, width = image_shape
    short_side = float(min(height, width))
    cluster_side = max(anchor.width, anchor.height) * float(context_factor)
    base_side = max(short_side * float(base_slice_fraction), cluster_side)
    side = base_side / max(float(zoom), 1e-6)
    side = float(
        np.clip(
            side,
            short_side * float(min_slice_fraction),
            short_side * float(max_slice_fraction),
        )
    )
    roi = np.asarray(
        [
            anchor.cx - side / 2.0,
            anchor.cy - side / 2.0,
            anchor.cx + side / 2.0,
            anchor.cy + side / 2.0,
        ],
        dtype=np.float32,
    )
    # Shift the square at image boundaries instead of shrinking it.
    if roi[0] < 0:
        roi[[0, 2]] -= roi[0]
    if roi[1] < 0:
        roi[[1, 3]] -= roi[1]
    if roi[2] > width:
        roi[[0, 2]] -= roi[2] - width
    if roi[3] > height:
        roi[[1, 3]] -= roi[3] - height
    return clip_box(roi, image_shape)


def union_area_ratio(rois: list[np.ndarray], image_shape: tuple[int, int], grid: int = 32) -> float:
    if not rois:
        return 0.0
    height, width = image_shape
    mask = np.zeros((grid, grid), dtype=np.uint8)
    for roi in rois:
        x1 = int(np.floor(float(roi[0]) / max(width, 1) * grid))
        y1 = int(np.floor(float(roi[1]) / max(height, 1) * grid))
        x2 = int(np.ceil(float(roi[2]) / max(width, 1) * grid))
        y2 = int(np.ceil(float(roi[3]) / max(height, 1) * grid))
        mask[max(0, y1) : min(grid, y2), max(0, x1) : min(grid, x2)] = 1
    return float(mask.mean())
