from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..core.geometry import iou_matrix
from ..core.types import Detections


class AP50Accumulator:
    def __init__(self, classes: tuple[int, ...], iou_threshold: float = 0.5) -> None:
        self.classes = tuple(int(value) for value in classes)
        self.iou_threshold = float(iou_threshold)
        self.records: dict[int, list[tuple[float, int]]] = defaultdict(list)
        self.gt_count: dict[int, int] = defaultdict(int)

    def update(self, predictions: Detections, ground_truth: Detections) -> None:
        for class_id in self.classes:
            pred_idx = np.flatnonzero(predictions.classes == class_id)
            gt_idx = np.flatnonzero(ground_truth.classes == class_id)
            self.gt_count[class_id] += int(len(gt_idx))
            matched = np.zeros((len(gt_idx),), dtype=bool)
            order = pred_idx[np.argsort(-predictions.scores[pred_idx])]
            for pred_index in order:
                is_true_positive = 0
                available = np.flatnonzero(~matched)
                if len(available):
                    overlaps = iou_matrix(
                        predictions.boxes[[pred_index]],
                        ground_truth.boxes[gt_idx[available]],
                    )[0]
                    best = int(np.argmax(overlaps))
                    if float(overlaps[best]) >= self.iou_threshold:
                        matched[available[best]] = True
                        is_true_positive = 1
                self.records[class_id].append(
                    (float(predictions.scores[pred_index]), is_true_positive)
                )

    def compute(self) -> dict:
        class_ap: dict[str, float] = {}
        total_tp = 0
        total_fp = 0
        total_gt = sum(self.gt_count.values())
        for class_id in self.classes:
            records = sorted(self.records[class_id], key=lambda item: item[0], reverse=True)
            if not records:
                if self.gt_count[class_id] > 0:
                    class_ap[str(class_id)] = 0.0
                continue
            tp = np.cumsum([item[1] for item in records], dtype=np.float64)
            fp = np.cumsum([1 - item[1] for item in records], dtype=np.float64)
            total_tp += int(tp[-1])
            total_fp += int(fp[-1])
            if self.gt_count[class_id] == 0:
                continue
            recall = tp / max(self.gt_count[class_id], 1)
            precision = tp / np.maximum(tp + fp, 1e-9)
            sampled = []
            for threshold in np.linspace(0.0, 1.0, 101):
                valid = precision[recall >= threshold]
                sampled.append(float(valid.max()) if len(valid) else 0.0)
            class_ap[str(class_id)] = float(np.mean(sampled))
        return {
            "ap50": float(np.mean(list(class_ap.values()))) if class_ap else 0.0,
            "precision": float(total_tp / max(total_tp + total_fp, 1)),
            "recall": float(total_tp / max(total_gt, 1)),
            "true_positives": int(total_tp),
            "false_positives": int(total_fp),
            "ground_truth": int(total_gt),
            "ap50_by_class": class_ap,
        }
