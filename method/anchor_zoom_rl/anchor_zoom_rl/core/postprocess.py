from __future__ import annotations

import numpy as np

from .geometry import iou_matrix
from .types import Detections


def class_aware_nms(detections: Detections, iou_threshold: float, max_detections: int) -> Detections:
    if len(detections) == 0:
        return detections
    kept: list[int] = []
    for class_id in np.unique(detections.classes):
        class_indices = np.flatnonzero(detections.classes == class_id)
        order = class_indices[np.argsort(-detections.scores[class_indices])]
        while len(order):
            current = int(order[0])
            kept.append(current)
            if len(order) == 1:
                break
            overlaps = iou_matrix(detections.boxes[[current]], detections.boxes[order[1:]])[0]
            order = order[1:][overlaps <= float(iou_threshold)]
    kept.sort(key=lambda index: float(detections.scores[index]), reverse=True)
    return detections.take(kept[: int(max_detections)])


def merge_detections(
    current: Detections,
    crop: Detections,
    iou_threshold: float,
    max_detections: int,
) -> Detections:
    return class_aware_nms(
        Detections.concatenate(current, crop),
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )


def crop_utility(crop: Detections, current: Detections, duplicate_iou: float) -> float:
    if len(crop) == 0:
        return 0.0
    novel = np.ones((len(crop),), dtype=bool)
    for class_id in np.unique(crop.classes):
        crop_indices = np.flatnonzero(crop.classes == class_id)
        current_indices = np.flatnonzero(current.classes == class_id)
        if len(current_indices) == 0:
            continue
        overlaps = iou_matrix(crop.boxes[crop_indices], current.boxes[current_indices])
        novel[crop_indices] = overlaps.max(axis=1) < float(duplicate_iou)
    if not novel.any():
        return 0.0
    return float(np.sum(crop.scores[novel]) / max(np.sqrt(float(novel.sum())), 1.0))


def translate_crop_detections(crop: Detections, roi: np.ndarray) -> Detections:
    if len(crop) == 0:
        return crop
    boxes = crop.boxes.copy()
    boxes[:, [0, 2]] += float(roi[0])
    boxes[:, [1, 3]] += float(roi[1])
    return Detections(boxes, crop.scores, crop.classes)
