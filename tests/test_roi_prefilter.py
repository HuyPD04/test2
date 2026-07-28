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

    def test_high_confidence_region_is_penalized(self) -> None:
        det = DetectionCache(
            image_path="image.jpg",
            image_shape=(100, 100),
            boxes=np.asarray(
                [[10, 10, 30, 30], [70, 70, 80, 80]],
                dtype=np.float32,
            ),
            scores=np.asarray([0.9, 0.25], dtype=np.float32),
            classes=np.asarray([0, 0], dtype=np.float32),
            feature=np.zeros((1,), dtype=np.float32),
            feature_layers=(16,),
            objectness_map=np.ones((1, 10, 10), dtype=np.float32) * 0.5,
            spatial_feature_map=np.zeros((4, 10, 10), dtype=np.float32),
        )
        rois = [
            np.asarray([0, 0, 45, 45], dtype=np.float32),
            np.asarray([55, 55, 100, 100], dtype=np.float32),
        ]

        scores = score_roi_candidates(
            det,
            rois,
            StateConfig(grid_size=10),
            (),
            ClassMapping(),
            high_conf_penalty=1.0,
        )

        self.assertGreater(float(scores[1]), float(scores[0]))

    def test_spatial_nms_skips_overlapping_candidate(self) -> None:
        rois = [
            np.asarray([0, 0, 50, 50], dtype=np.float32),
            np.asarray([5, 5, 52, 52], dtype=np.float32),
            np.asarray([60, 60, 100, 100], dtype=np.float32),
        ]

        selected = select_roi_candidates(
            np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
            topk=2,
            rois=rois,
            overlap_threshold=0.35,
        )

        self.assertEqual(selected, [0, 2])


if __name__ == "__main__":
    unittest.main()
