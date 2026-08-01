from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.nms import batched_nms_numpy, nms_numpy
from rl_sahi.inference.merge import class_aware_nms


class NmsTest(unittest.TestCase):
    def test_cdn_uses_cluster_diou_suppression(self) -> None:
        boxes = np.asarray(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
                [2.0, 2.0, 12.0, 12.0],
            ],
            dtype=np.float32,
        )
        scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

        keep = nms_numpy(boxes, scores, 0.5, nms_type="cdn")

        self.assertEqual(keep.tolist(), [0, 2])

    def test_unknown_nms_type_raises(self) -> None:
        boxes = np.asarray([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32)
        scores = np.asarray([0.9], dtype=np.float32)

        with self.assertRaises(ValueError):
            nms_numpy(boxes, scores, 0.5, nms_type="not-a-real-nms")

    def test_batched_nms_keeps_overlapping_boxes_from_different_classes(self) -> None:
        boxes = np.asarray(
            [
                [0.0, 0.0, 10.0, 10.0],
                [1.0, 1.0, 11.0, 11.0],
                [0.0, 0.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )
        scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
        classes = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

        keep = batched_nms_numpy(boxes, scores, classes, 0.5)
        merge_keep = class_aware_nms(boxes, scores, classes, 0.5, nms_type="standard")

        self.assertEqual(keep.tolist(), [0, 2])
        self.assertEqual(merge_keep.tolist(), [0, 2])


if __name__ == "__main__":
    unittest.main()
