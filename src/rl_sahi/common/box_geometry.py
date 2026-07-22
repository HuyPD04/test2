from __future__ import annotations

import numpy as np

from rl_sahi.common.box_types import EPS, as_boxes


def area(boxes: np.ndarray) -> np.ndarray:
    boxes = as_boxes(boxes)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])


def intersection_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = as_boxes(a)
    b = as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return (np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)).astype(np.float32)


def covered_area_by_boxes(targets: np.ndarray, covers: np.ndarray) -> np.ndarray:
    targets = as_boxes(targets)
    covers = as_boxes(covers)
    output = np.zeros((len(targets),), dtype=np.float32)
    if len(targets) == 0 or len(covers) == 0:
        return output
    for index, target in enumerate(targets):
        x1 = np.maximum(target[0], covers[:, 0])
        y1 = np.maximum(target[1], covers[:, 1])
        x2 = np.minimum(target[2], covers[:, 2])
        y2 = np.minimum(target[3], covers[:, 3])
        valid = (x2 > x1) & (y2 > y1)
        if valid.any():
            output[index] = float(_union_area(np.stack((x1[valid], y1[valid], x2[valid], y2[valid]), axis=1)))
    return output


def _union_area(boxes: np.ndarray) -> float:
    boxes = as_boxes(boxes)
    if len(boxes) == 0:
        return 0.0
    xs = np.unique(np.concatenate((boxes[:, 0], boxes[:, 2]), axis=0))
    if len(xs) < 2:
        return 0.0
    total = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        width = float(right - left)
        if width <= 0.0:
            continue
        active = boxes[(boxes[:, 0] < right) & (boxes[:, 2] > left)]
        if len(active) == 0:
            continue
        intervals = sorted((float(y1), float(y2)) for y1, y2 in active[:, [1, 3]])
        merged_height = 0.0
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_height += max(current_end - current_start, 0.0)
                current_start, current_end = start, end
        merged_height += max(current_end - current_start, 0.0)
        total += width * merged_height
    return total


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = intersection_matrix(a, b)
    if inter.size == 0:
        return inter
    area_a = area(a)[:, None]
    area_b = area(b)[None, :]
    return (inter / np.maximum(area_a + area_b - inter, EPS)).astype(np.float32)


def ioa_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over area of b, useful for "does ROI cover object?" tests."""
    inter = intersection_matrix(a, b)
    if inter.size == 0:
        return inter
    area_b = area(b)[None, :]
    return (inter / np.maximum(area_b, EPS)).astype(np.float32)


def centers(boxes: np.ndarray) -> np.ndarray:
    boxes = as_boxes(boxes)
    if boxes.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0], axis=1)


def center_inside(roi: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    roi = as_boxes(roi).reshape(1, 4)[0]
    pts = centers(boxes)
    if len(pts) == 0:
        return np.zeros((0,), dtype=bool)
    return (pts[:, 0] >= roi[0]) & (pts[:, 0] <= roi[2]) & (pts[:, 1] >= roi[1]) & (pts[:, 1] <= roi[3])


def normalized_box(box: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    b = as_boxes(box).reshape(1, 4)[0]
    return np.array([b[0] / w, b[1] / h, b[2] / w, b[3] / h], dtype=np.float32)
