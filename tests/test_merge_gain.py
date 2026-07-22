from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.inference.merge import (
    accepts_novel_detections,
    merge_predictions,
    new_detection_gain_after_merge,
    new_detection_stats_after_merge,
    new_detection_utility_after_merge,
)


class MergeGainTest(unittest.TestCase):
    def test_replacement_of_existing_same_class_box_is_not_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.4], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32),
            candidate_scores=np.array([0.9], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
        )

        self.assertEqual(gain, 0)

    def test_shifted_replacement_of_existing_same_class_box_is_not_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.4], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[12.0, 12.0, 32.0, 32.0]], dtype=np.float32),
            candidate_scores=np.array([0.9], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
            duplicate_iou=0.5,
        )

        self.assertEqual(gain, 0)

    def test_spatially_new_same_class_box_counts_as_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.9], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[60.0, 60.0, 80.0, 80.0]], dtype=np.float32),
            candidate_scores=np.array([0.8], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
        )

        self.assertEqual(gain, 1)

    def test_spatially_new_utility_uses_candidate_confidence(self) -> None:
        utility = new_detection_utility_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.9], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[60.0, 60.0, 80.0, 80.0]], dtype=np.float32),
            candidate_scores=np.array([0.8], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
        )

        self.assertAlmostEqual(utility, 0.8, places=6)

    def test_same_location_different_class_counts_as_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.9], dtype=np.float32)],
            previous_classes_parts=[np.array([0.0], dtype=np.float32)],
            candidate_boxes=np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32),
            candidate_scores=np.array([0.8], dtype=np.float32),
            candidate_classes=np.array([2.0], dtype=np.float32),
        )

        self.assertEqual(gain, 1)

    def test_same_vehicle_location_different_class_lower_score_is_not_new(self) -> None:
        gain = new_detection_gain_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            previous_scores_parts=[np.array([0.9], dtype=np.float32)],
            previous_classes_parts=[np.array([4.0], dtype=np.float32)],
            candidate_boxes=np.array([[10.0, 10.0, 30.0, 30.0]], dtype=np.float32),
            candidate_scores=np.array([0.8], dtype=np.float32),
            candidate_classes=np.array([3.0], dtype=np.float32),
        )

        self.assertEqual(gain, 0)

    def test_vehicle_cross_class_duplicate_keeps_highest_score(self) -> None:
        boxes, scores, classes = merge_predictions(
            image_shape=(100, 100),
            merge_iou=0.5,
            boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0], [10.0, 10.0, 30.0, 30.0]], dtype=np.float32)],
            scores_parts=[np.array([0.38, 0.51], dtype=np.float32)],
            classes_parts=[np.array([3.0, 4.0], dtype=np.float32)],
        )

        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(float(scores[0]), 0.51, places=6)
        self.assertEqual(int(classes[0]), 4)

    def test_vehicle_cross_class_containment_keeps_highest_score(self) -> None:
        boxes, scores, classes = merge_predictions(
            image_shape=(100, 100),
            merge_iou=0.5,
            boxes_parts=[np.array([[10.0, 10.0, 30.0, 30.0], [9.0, 9.0, 31.0, 31.0]], dtype=np.float32)],
            scores_parts=[np.array([0.53, 0.38], dtype=np.float32)],
            classes_parts=[np.array([3.0, 4.0], dtype=np.float32)],
        )

        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(float(scores[0]), 0.53, places=6)
        self.assertEqual(int(classes[0]), 3)

    def test_single_novel_detection_uses_score_gate_not_sum_gate(self) -> None:
        gain, utility, max_score = new_detection_stats_after_merge(
            image_shape=(100, 100),
            merge_iou=0.5,
            previous_boxes_parts=[],
            previous_scores_parts=[],
            previous_classes_parts=[],
            candidate_boxes=np.array([[60.0, 60.0, 80.0, 80.0]], dtype=np.float32),
            candidate_scores=np.array([0.6], dtype=np.float32),
            candidate_classes=np.array([0.0], dtype=np.float32),
        )
        self.assertTrue(accepts_novel_detections(gain, utility, max_score, 1, 0.8, 0.45))

    def test_multiple_weak_detections_do_not_pass_score_gate(self) -> None:
        self.assertFalse(accepts_novel_detections(2, 0.8, 0.4, 1, 0.8, 0.45))


if __name__ == "__main__":
    unittest.main()
