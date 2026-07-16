from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.cache import DetectionCache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.inference.roi_prefilter import score_roi_candidates, select_roi_candidates
from rl_sahi.rl.state_config import StateConfig


class RoiPrefilterTest(unittest.TestCase):
    def test_objectness_selects_highest_scoring_rois(self) -> None:
        objectness = np.zeros((10, 10), dtype=np.float32)
        objectness[:, :3] = 0.1
        objectness[:, 3:6] = 1.0
        objectness[:, 6:] = 0.5
        det = DetectionCache(
            image_path="image.jpg",
            image_shape=(100, 100),
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            classes=np.zeros((0,), dtype=np.float32),
            feature=np.zeros((1,), dtype=np.float32),
            feature_layers=(16,),
            objectness_map=objectness,
            spatial_feature_map=np.zeros((4, 10, 10), dtype=np.float32),
        )
        rois = [
            np.asarray([0, 0, 30, 100], dtype=np.float32),
            np.asarray([30, 0, 60, 100], dtype=np.float32),
            np.asarray([60, 0, 100, 100], dtype=np.float32),
        ]

        scores = score_roi_candidates(det, rois, StateConfig(), (), ClassMapping())

        self.assertGreater(float(scores[1]), float(scores[2]))
        self.assertGreater(float(scores[2]), float(scores[0]))
        self.assertEqual(select_roi_candidates(scores, topk=2), [1, 2])

    def test_non_positive_topk_keeps_all_candidates(self) -> None:
        self.assertEqual(select_roi_candidates(np.asarray([0.2, 0.1]), topk=0), [0, 1])


if __name__ == "__main__":
    unittest.main()
