from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.eval.benchmark import _precision_recall_at_iou, select_benchmark_images
from rl_sahi.common.config import load_default_config


class BenchmarkSamplingTest(unittest.TestCase):
    def test_project_target_classes_are_explicit(self) -> None:
        cfg = load_default_config(None, ROOT)
        self.assertEqual(cfg.target_classes(), (0, 2, 3, 5, 8, 9))

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


if __name__ == "__main__":
    unittest.main()
