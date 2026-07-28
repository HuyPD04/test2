from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.pipeline import _slice_predictions_for_merge, filter_boundary_boxes


class InferencePostprocessTest(unittest.TestCase):
    def test_boundary_margin_controls_internal_edge_filtering(self) -> None:
        boxes = np.asarray([[14.0, 20.0, 25.0, 35.0]], dtype=np.float32)
        scores = np.asarray([0.9], dtype=np.float32)
        classes = np.asarray([0.0], dtype=np.float32)
        roi = np.asarray([10.0, 10.0, 50.0, 50.0], dtype=np.float32)

        kept = filter_boundary_boxes(boxes, scores, classes, roi, (100, 100), margin=2.0)
        removed = filter_boundary_boxes(boxes, scores, classes, roi, (100, 100), margin=6.0)

        self.assertEqual(len(kept[0]), 1)
        self.assertEqual(len(removed[0]), 0)

    def test_append_novel_only_excludes_same_class_duplicate(self) -> None:
        full_boxes = np.asarray([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)
        full_scores = np.asarray([0.9], dtype=np.float32)
        full_classes = np.asarray([0.0], dtype=np.float32)
        candidate_boxes = np.asarray(
            [[10.0, 10.0, 30.0, 30.0], [60.0, 60.0, 80.0, 80.0]],
            dtype=np.float32,
        )
        candidate_scores = np.asarray([0.8, 0.7], dtype=np.float32)
        candidate_classes = np.asarray([0.0, 0.0], dtype=np.float32)
        cfg = InferenceConfig(
            merge_iou=0.5,
            duplicate_iou=0.5,
            cross_class_duplicate_iou=None,
            cross_class_duplicate_ios=None,
            append_novel_only=True,
            nms_type="standard",
        )

        boxes, scores, classes, reliability = _slice_predictions_for_merge(
            full_boxes,
            full_scores,
            full_classes,
            [],
            [],
            [],
            [],
            candidate_boxes,
            candidate_scores,
            candidate_classes,
            np.asarray([0.5, 0.5], dtype=np.float32),
            np.asarray([50.0, 50.0, 90.0, 90.0], dtype=np.float32),
            (100, 100),
            {},
            cfg,
        )

        np.testing.assert_allclose(boxes, candidate_boxes[1:])
        np.testing.assert_allclose(scores, candidate_scores[1:])
        np.testing.assert_allclose(classes, candidate_classes[1:])
        self.assertEqual(len(reliability), 1)


if __name__ == "__main__":
    unittest.main()
