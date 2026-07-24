from __future__ import annotations

import numpy as np

from anchor_zoom_rl.core.postprocess import crop_reliability, merge_detections
from anchor_zoom_rl.core.types import Detections
from anchor_zoom_rl.runtime.metrics import AP50Accumulator


def test_merge_is_class_aware_and_keeps_higher_score_duplicate() -> None:
    current = Detections(
        np.asarray([[10, 10, 20, 20]], dtype=np.float32),
        np.asarray([0.7], dtype=np.float32),
        np.asarray([0], dtype=np.int64),
    )
    crop = Detections(
        np.asarray([[10, 10, 20, 20], [10, 10, 20, 20]], dtype=np.float32),
        np.asarray([0.9, 0.8], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64),
    )
    merged = merge_detections(current, crop, iou_threshold=0.5, max_detections=10)
    assert len(merged) == 2
    assert sorted(merged.classes.tolist()) == [0, 1]
    assert float(merged.scores[merged.classes == 0][0]) > 0.89


def test_cross_class_cleanup_suppresses_lower_score_duplicate() -> None:
    detections = Detections(
        np.asarray([[10, 10, 20, 20], [10, 10, 20, 20]], dtype=np.float32),
        np.asarray([0.9, 0.7], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int64),
    )
    merged = merge_detections(
        Detections.empty(),
        detections,
        iou_threshold=0.5,
        max_detections=10,
        cross_class_iou=0.85,
        cross_class_ios=0.95,
    )
    assert len(merged) == 1
    assert merged.classes.tolist() == [0]


def test_crop_reliability_penalizes_boundary_noise() -> None:
    current = Detections.empty()
    centered = Detections(
        np.asarray([[30, 30, 50, 50]], dtype=np.float32),
        np.asarray([0.8], dtype=np.float32),
        np.asarray([0], dtype=np.int64),
    )
    boundary = Detections(
        np.asarray([[0, 30, 20, 50]], dtype=np.float32),
        np.asarray([0.3], dtype=np.float32),
        np.asarray([0], dtype=np.int64),
    )
    roi = np.asarray([0, 0, 100, 100], dtype=np.float32)
    good = crop_reliability(centered, current, roi, 0.5, 0.0, 0.5)
    poor = crop_reliability(boundary, current, roi, 0.1, 0.0, 0.5)
    assert good > 0.7
    assert poor < 0.35


def test_ap50_accumulator_reports_perfect_prediction() -> None:
    detections = Detections(
        np.asarray([[10, 10, 20, 20]], dtype=np.float32),
        np.asarray([0.9], dtype=np.float32),
        np.asarray([0], dtype=np.int64),
    )
    metric = AP50Accumulator((0,), iou_threshold=0.5)
    metric.update(detections, detections)
    result = metric.compute()
    assert result["ap50"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
