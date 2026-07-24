from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import iou_matrix
from .types import Detections


@dataclass(slots=True)
class HardRegionData:
    gt_boxes: np.ndarray
    gt_classes: np.ndarray
    hard_mask: np.ndarray
    best_iou: np.ndarray
    best_score: np.ndarray

    def __post_init__(self) -> None:
        self.gt_boxes = np.asarray(self.gt_boxes, dtype=np.float32).reshape(-1, 4)
        self.gt_classes = np.asarray(self.gt_classes, dtype=np.int64).reshape(-1)
        self.hard_mask = np.asarray(self.hard_mask, dtype=bool).reshape(-1)
        self.best_iou = np.asarray(self.best_iou, dtype=np.float32).reshape(-1)
        self.best_score = np.asarray(self.best_score, dtype=np.float32).reshape(-1)
        size = len(self.gt_boxes)
        if not all(
            len(values) == size
            for values in (
                self.gt_classes,
                self.hard_mask,
                self.best_iou,
                self.best_score,
            )
        ):
            raise ValueError("Hard-region arrays must have the same length")

    @property
    def hard_boxes(self) -> np.ndarray:
        return self.gt_boxes[self.hard_mask]

    @property
    def hard_classes(self) -> np.ndarray:
        return self.gt_classes[self.hard_mask]


def build_hard_regions(
    full_detections: Detections,
    ground_truth: Detections,
    match_iou: float,
    low_confidence: float,
) -> HardRegionData:
    count = len(ground_truth)
    best_iou = np.zeros((count,), dtype=np.float32)
    best_score = np.zeros((count,), dtype=np.float32)
    for gt_index in range(count):
        candidates = np.flatnonzero(
            full_detections.classes == ground_truth.classes[gt_index]
        )
        if len(candidates) == 0:
            continue
        overlaps = iou_matrix(
            ground_truth.boxes[[gt_index]],
            full_detections.boxes[candidates],
        )[0]
        best_local = int(np.argmax(overlaps))
        best_iou[gt_index] = float(overlaps[best_local])
        best_score[gt_index] = float(full_detections.scores[candidates[best_local]])
    hard_mask = (best_iou < float(match_iou)) | (
        best_score < float(low_confidence)
    )
    return HardRegionData(
        gt_boxes=ground_truth.boxes.copy(),
        gt_classes=ground_truth.classes.copy(),
        hard_mask=hard_mask,
        best_iou=best_iou,
        best_score=best_score,
    )
