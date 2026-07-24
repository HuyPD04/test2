from __future__ import annotations

import numpy as np

from anchor_zoom_rl.core.postprocess import merge_detections
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
