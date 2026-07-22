from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.nms import nms_numpy


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


if __name__ == "__main__":
    unittest.main()
