from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.eval.benchmark import (
    _ap_from_pr,
    _effective_warmup_images,
    _evaluate_method,
    _precision_recall_at_iou,
    select_benchmark_images,
)
from rl_sahi.common.config import load_default_config


class BenchmarkSamplingTest(unittest.TestCase):
    def test_project_target_classes_are_explicit(self) -> None:
        cfg = load_default_config(None, ROOT)
        self.assertEqual(cfg.target_classes(), tuple(range(10)))

    def test_stratified_sampling_covers_sequences_before_repeating(self) -> None:
        images = [
            Path(f"s{sequence}_frame_{frame}.jpg")
            for sequence in range(4)
            for frame in range(3)
        ]
        selected = select_benchmark_images(images, limit=4, sampling="stratified", seed=7)
        sequence_ids = {image.stem.split("_", 1)[0] for image in selected}
        self.assertEqual(len(sequence_ids), 4)

    def test_stratified_sampling_is_deterministic(self) -> None:
        images = [Path(f"{sequence:04d}_{frame:04d}.jpg") for sequence in range(5) for frame in range(4)]
        first = select_benchmark_images(images, limit=11, sampling="stratified", seed=42)
        second = select_benchmark_images(images, limit=11, sampling="stratified", seed=42)
        self.assertEqual(first, second)

    def test_warmup_uses_ten_images_by_default_for_larger_runs(self) -> None:
        self.assertEqual(_effective_warmup_images(100, 10), 10)

    def test_warmup_always_leaves_one_timed_image(self) -> None:
        self.assertEqual(_effective_warmup_images(5, 10), 4)
        self.assertEqual(_effective_warmup_images(1, 10), 0)

    def test_precision_recall_are_micro_averaged_at_iou(self) -> None:
        ground_truth = {
            "a": (
                np.asarray([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
                np.asarray([0, 1], dtype=np.float32),
                (40, 40),
            )
        }
        predictions = {
            "a": (
                np.asarray(
                    [[0, 0, 10, 10], [20, 20, 30, 30], [32, 32, 39, 39]],
                    dtype=np.float32,
                ),
                np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
                np.asarray([0, 1, 0], dtype=np.float32),
            )
        }
        precision, recall = _precision_recall_at_iou(
            predictions, ground_truth, (0, 1), 0.5
        )
        self.assertAlmostEqual(precision, 2.0 / 3.0)
        self.assertAlmostEqual(recall, 1.0)

    def test_ap_uses_coco_101_point_interpolation(self) -> None:
        ap = _ap_from_pr(
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.0, 1.0], dtype=np.float32),
            total_gt=2,
        )
        self.assertAlmostEqual(ap, 51.0 / 101.0, places=6)

    def test_eval_max_detections_is_applied_per_image_and_class(self) -> None:
        ground_truth = {
            "a": (
                np.asarray([[0, 0, 10, 10]], dtype=np.float32),
                np.asarray([0], dtype=np.float32),
                (40, 40),
            )
        }
        predictions = {
            "a": (
                np.asarray([[30, 30, 39, 39], [0, 0, 10, 10]], dtype=np.float32),
                np.asarray([0.99, 0.50], dtype=np.float32),
                np.asarray([1, 0], dtype=np.float32),
            )
        }
        metrics = _evaluate_method(
            predictions,
            ground_truth,
            target_classes=(0, 1),
            iou_threshold=0.5,
            small_area_threshold=1.0,
            max_detections=1,
        )
        self.assertAlmostEqual(metrics["AP50"], 1.0)
        self.assertEqual(metrics["eval_max_detections"], 1.0)


if __name__ == "__main__":
    unittest.main()
