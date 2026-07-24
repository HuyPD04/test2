from __future__ import annotations

import numpy as np

from .geometry import ios_matrix, iou_matrix
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
    cross_class_iou: float | None = None,
    cross_class_ios: float | None = None,
    cross_class_score_ratio: float = 1.0,
    cross_class_groups: tuple[tuple[int, ...], ...] | None = None,
) -> Detections:
    merged = class_aware_nms(
        Detections.concatenate(current, crop),
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
    return cross_class_duplicate_cleanup(
        merged,
        iou_threshold=cross_class_iou,
        ios_threshold=cross_class_ios,
        max_detections=max_detections,
        score_ratio=cross_class_score_ratio,
        class_groups=cross_class_groups,
    )


def cross_class_duplicate_cleanup(
    detections: Detections,
    iou_threshold: float | None,
    ios_threshold: float | None,
    max_detections: int,
    score_ratio: float = 1.0,
    class_groups: tuple[tuple[int, ...], ...] | None = None,
) -> Detections:
    if len(detections) <= 1 or (iou_threshold is None and ios_threshold is None):
        return detections.take(np.argsort(-detections.scores)[: int(max_detections)])
    order = np.argsort(-detections.scores)
    kept: list[int] = []
    for index in order:
        if not kept:
            kept.append(int(index))
            continue
        kept_array = np.asarray(kept, dtype=np.int64)
        different_class = detections.classes[kept_array] != detections.classes[index]
        if class_groups:
            candidate_class = int(detections.classes[index])
            confusable = np.asarray(
                [
                    any(
                        candidate_class in group
                        and int(detections.classes[kept_index]) in group
                        for group in class_groups
                    )
                    for kept_index in kept_array
                ],
                dtype=bool,
            )
            different_class &= confusable
        if not different_class.any():
            kept.append(int(index))
            continue
        compare = kept_array[different_class]
        compare = compare[
            detections.scores[index]
            <= detections.scores[compare] * float(score_ratio)
        ]
        if len(compare) == 0:
            kept.append(int(index))
            continue
        duplicate = False
        if iou_threshold is not None:
            overlap = iou_matrix(
                detections.boxes[[index]], detections.boxes[compare]
            )[0]
            duplicate = bool(np.any(overlap >= float(iou_threshold)))
        if not duplicate and ios_threshold is not None:
            overlap = ios_matrix(
                detections.boxes[[index]], detections.boxes[compare]
            )[0]
            duplicate = bool(np.any(overlap >= float(ios_threshold)))
        if not duplicate:
            kept.append(int(index))
        if len(kept) >= int(max_detections):
            break
    return detections.take(kept)


def crop_reliability(
    crop: Detections,
    current: Detections,
    roi: np.ndarray,
    anchor_score: float,
    history_overlap: float,
    duplicate_iou: float,
    boundary_margin: float = 0.04,
) -> float:
    novel = novel_detection_mask(crop, current, duplicate_iou)
    if not novel.any():
        return 0.0
    novel_boxes = crop.boxes[novel]
    novel_scores = crop.scores[novel]
    roi = np.asarray(roi, dtype=np.float32)
    width = max(float(roi[2] - roi[0]), 1.0)
    height = max(float(roi[3] - roi[1]), 1.0)
    margin_x = width * float(boundary_margin)
    margin_y = height * float(boundary_margin)
    boundary = (
        (novel_boxes[:, 0] <= roi[0] + margin_x)
        | (novel_boxes[:, 1] <= roi[1] + margin_y)
        | (novel_boxes[:, 2] >= roi[2] - margin_x)
        | (novel_boxes[:, 3] >= roi[3] - margin_y)
    )
    reliability = (
        0.55 * float(novel_scores.mean())
        + 0.20 * float(novel_scores.max())
        + 0.15 * float(np.clip(anchor_score, 0.0, 1.0))
        + 0.10 * (1.0 - float(np.clip(history_overlap, 0.0, 1.0)))
        - 0.25 * float(boundary.mean())
    )
    return float(np.clip(reliability, 0.0, 1.0))


def crop_utility(crop: Detections, current: Detections, duplicate_iou: float) -> float:
    if len(crop) == 0:
        return 0.0
    novel = novel_detection_mask(crop, current, duplicate_iou)
    if not novel.any():
        return 0.0
    return float(np.sum(crop.scores[novel]) / max(np.sqrt(float(novel.sum())), 1.0))


def novel_detection_mask(
    crop: Detections,
    current: Detections,
    duplicate_iou: float,
) -> np.ndarray:
    novel = np.ones((len(crop),), dtype=bool)
    for class_id in np.unique(crop.classes):
        crop_indices = np.flatnonzero(crop.classes == class_id)
        current_indices = np.flatnonzero(current.classes == class_id)
        if len(current_indices) == 0:
            continue
        overlaps = iou_matrix(crop.boxes[crop_indices], current.boxes[current_indices])
        novel[crop_indices] = overlaps.max(axis=1) < float(duplicate_iou)
    return novel


def translate_crop_detections(crop: Detections, roi: np.ndarray) -> Detections:
    if len(crop) == 0:
        return crop
    boxes = crop.boxes.copy()
    boxes[:, [0, 2]] += float(roi[0])
    boxes[:, [1, 3]] += float(roi[1])
    return Detections(boxes, crop.scores, crop.classes)
