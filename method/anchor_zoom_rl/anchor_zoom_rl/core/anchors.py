from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..config import AnchorConfig
from .geometry import box_area
from .types import Anchor, Detections


def generate_anchors(
    detections: Detections,
    image_shape: tuple[int, int],
    cfg: AnchorConfig,
) -> list[Anchor]:
    height, width = image_shape
    image_area = float(max(height * width, 1))
    keep = detections.scores >= float(cfg.min_confidence)
    boxes = detections.boxes[keep]
    scores = detections.scores[keep]
    areas = box_area(boxes) / image_area
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5 if len(boxes) else np.zeros((0, 2))

    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, center in enumerate(centers):
        gx = min(int(center[0] / max(width, 1) * cfg.cluster_grid_size), cfg.cluster_grid_size - 1)
        gy = min(int(center[1] / max(height, 1) * cfg.cluster_grid_size), cfg.cluster_grid_size - 1)
        groups[(gx, gy)].append(index)

    anchors: list[Anchor] = []
    for indices in groups.values():
        idx = np.asarray(indices, dtype=np.int64)
        group_boxes = boxes[idx]
        group_scores = scores[idx]
        group_areas = areas[idx]
        weights = np.maximum(1.0 - group_scores, 0.05)
        center = np.average(centers[idx], axis=0, weights=weights)
        cluster_box = np.asarray(
            [
                group_boxes[:, 0].min(),
                group_boxes[:, 1].min(),
                group_boxes[:, 2].max(),
                group_boxes[:, 3].max(),
            ],
            dtype=np.float32,
        )
        small_fraction = float(np.mean(group_areas <= cfg.small_area_ratio))
        uncertainty = float(np.mean(1.0 - group_scores))
        density = min(len(idx) / max(cfg.density_norm, 1.0), 1.0)
        score = (
            cfg.uncertainty_weight * uncertainty
            + cfg.density_weight * density
            + cfg.small_object_weight * small_fraction
        )
        anchors.append(
            Anchor(
                cx=float(center[0]),
                cy=float(center[1]),
                width=float(cluster_box[2] - cluster_box[0]),
                height=float(cluster_box[3] - cluster_box[1]),
                score=float(score),
                count=int(len(idx)),
                mean_conf=float(group_scores.mean()),
                max_conf=float(group_scores.max()),
                small_fraction=small_fraction,
                mean_area_ratio=float(group_areas.mean()),
            )
        )

    anchors.sort(key=lambda item: item.score, reverse=True)
    anchors = anchors[: cfg.top_k]
    if len(anchors) < cfg.top_k and cfg.add_fallback_grid:
        anchors.extend(_fallback_anchors(anchors, image_shape, cfg))
    return anchors[: cfg.top_k]


def _fallback_anchors(
    existing: list[Anchor],
    image_shape: tuple[int, int],
    cfg: AnchorConfig,
) -> list[Anchor]:
    height, width = image_shape
    candidates: list[Anchor] = []
    grid = max(int(cfg.fallback_grid_size), 1)
    min_distance = min(height, width) * cfg.base_slice_fraction * 0.5
    existing_centers = np.asarray([[a.cx, a.cy] for a in existing], dtype=np.float32)
    for gy in range(grid):
        for gx in range(grid):
            cx = (gx + 0.5) / grid * width
            cy = (gy + 0.5) / grid * height
            if len(existing_centers):
                distance = np.sqrt(np.sum((existing_centers - [cx, cy]) ** 2, axis=1))
                if float(distance.min()) < min_distance:
                    continue
            candidates.append(
                Anchor(
                    cx=float(cx),
                    cy=float(cy),
                    width=0.0,
                    height=0.0,
                    score=float(cfg.fallback_score),
                    count=0,
                    mean_conf=0.0,
                    max_conf=0.0,
                    small_fraction=0.0,
                    mean_area_ratio=0.0,
                    source="grid",
                )
            )
    candidates.sort(key=lambda item: (item.score, -item.cy, -item.cx), reverse=True)
    return candidates[: max(cfg.top_k - len(existing), 0)]
