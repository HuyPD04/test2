from __future__ import annotations

import numpy as np

from rl_sahi.common.boxes import area, centers, intersection_matrix
from rl_sahi.common.cache import DetectionCache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.rl.state_config import StateConfig


def _proposal_quality(scores: np.ndarray, cfg: StateConfig) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        return scores
    up = (scores - float(cfg.proposal_min_conf)) / max(
        float(cfg.proposal_peak_conf) - float(cfg.proposal_min_conf), 1e-6
    )
    down = (float(cfg.proposal_max_conf) - scores) / max(
        float(cfg.proposal_max_conf) - float(cfg.proposal_peak_conf), 1e-6
    )
    return np.clip(np.minimum(up, down), 0.0, 1.0).astype(np.float32)


def _objectness_score(det: DetectionCache, roi: np.ndarray) -> float:
    obj = np.asarray(det.objectness_map, dtype=np.float32)
    if obj.size == 0:
        return 0.0
    if obj.ndim == 2:
        heat = obj
    elif obj.ndim >= 3:
        heat = obj.reshape(-1, obj.shape[-2], obj.shape[-1]).max(axis=0)
    else:
        return 0.0
    grid_h, grid_w = heat.shape
    image_h, image_w = det.image_shape
    x1, y1, x2, y2 = np.asarray(roi, dtype=np.float32).reshape(4)
    gx1 = int(np.clip(np.floor(x1 / max(image_w, 1) * grid_w), 0, grid_w - 1))
    gy1 = int(np.clip(np.floor(y1 / max(image_h, 1) * grid_h), 0, grid_h - 1))
    gx2 = int(np.clip(np.ceil(x2 / max(image_w, 1) * grid_w), gx1 + 1, grid_w))
    gy2 = int(np.clip(np.ceil(y2 / max(image_h, 1) * grid_h), gy1 + 1, grid_h))
    window = heat[gy1:gy2, gx1:gx2]
    if window.size == 0:
        return 0.0
    return float(np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).mean())


def score_roi_candidates(
    det: DetectionCache,
    rois: list[np.ndarray],
    state_cfg: StateConfig,
    target_classes: tuple[int, ...],
    class_mapping: ClassMapping,
    high_conf_threshold: float = 0.5,
    high_conf_penalty: float = 1.0,
) -> np.ndarray:
    """Score candidate crops using signals already produced by full-image inference."""
    if not rois:
        return np.zeros((0,), dtype=np.float32)

    boxes = np.asarray(det.boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(det.scores, dtype=np.float32).reshape(-1)
    classes = class_mapping.map_model_classes(det.classes).astype(np.int64)
    class_mask = np.ones((len(classes),), dtype=bool)
    if target_classes:
        class_mask &= np.isin(classes, np.asarray(target_classes, dtype=np.int64))

    proposal_selection = class_mask & (
        scores >= float(state_cfg.proposal_min_conf)
    ) & (
        scores <= float(state_cfg.proposal_max_conf)
    )
    proposal_boxes = boxes[proposal_selection]
    proposal_scores = scores[proposal_selection]
    high_selection = class_mask & (scores >= float(high_conf_threshold))
    high_boxes = boxes[high_selection]
    high_scores = scores[high_selection]

    image_area = max(float(det.image_shape[0] * det.image_shape[1]), 1.0)
    if len(proposal_boxes):
        points = centers(proposal_boxes)
        weights = 0.25 + _proposal_quality(proposal_scores, state_cfg)
        weights += (
            area(proposal_boxes) / image_area <= float(state_cfg.small_area_ratio)
        ).astype(np.float32) * 0.75
    else:
        points = np.zeros((0, 2), dtype=np.float32)
        weights = np.zeros((0,), dtype=np.float32)
    high_points = (
        centers(high_boxes)
        if len(high_boxes)
        else np.zeros((0, 2), dtype=np.float32)
    )

    result = np.zeros((len(rois),), dtype=np.float32)
    for index, roi in enumerate(rois):
        roi = np.asarray(roi, dtype=np.float32).reshape(4)
        roi_ratio = max(float(area(roi.reshape(1, 4))[0]) / image_area, 1e-6)
        scale = 1.0 / max(np.sqrt(roi_ratio), 1e-3)
        if len(points):
            x1, y1, x2, y2 = roi
            inside = (
                (points[:, 0] >= x1)
                & (points[:, 0] <= x2)
                & (points[:, 1] >= y1)
                & (points[:, 1] <= y2)
            )
            if inside.any():
                result[index] += float(weights[inside].sum()) * scale
        if len(high_points) and float(high_conf_penalty) > 0.0:
            x1, y1, x2, y2 = roi
            high_inside = (
                (high_points[:, 0] >= x1)
                & (high_points[:, 0] <= x2)
                & (high_points[:, 1] >= y1)
                & (high_points[:, 1] <= y2)
            )
            if high_inside.any():
                result[index] -= (
                    float(high_conf_penalty)
                    * float(high_scores[high_inside].sum())
                    * scale
                )
        result[index] += 0.5 * _objectness_score(det, roi)
    return result


def select_roi_candidates(
    scores: np.ndarray,
    topk: int,
    rois: list[np.ndarray] | None = None,
    overlap_threshold: float = 0.35,
) -> list[int]:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        return []
    limit = len(scores) if int(topk) <= 0 else min(int(topk), len(scores))
    order = np.argsort(-scores, kind="stable")
    if rois is None:
        return sorted(int(index) for index in order[:limit])

    roi_array = np.asarray(rois, dtype=np.float32).reshape(-1, 4)
    if len(roi_array) != len(scores):
        raise ValueError("rois and scores must have the same length")
    selected: list[int] = []
    threshold = float(np.clip(overlap_threshold, 0.0, 1.0))
    roi_areas = np.maximum(area(roi_array), 1.0)
    for raw_index in order:
        index = int(raw_index)
        if selected:
            intersection = intersection_matrix(
                roi_array[index : index + 1],
                roi_array[selected],
            )[0]
            smaller_area = np.minimum(roi_areas[index], roi_areas[selected])
            overlap = intersection / np.maximum(smaller_area, 1.0)
            if bool(np.any(overlap >= threshold)):
                continue
        selected.append(index)
        if len(selected) >= limit:
            break
    return sorted(selected)
