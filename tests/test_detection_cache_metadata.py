from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.cache import (
    DetectionCache,
    HardRegionCache,
    detection_cache_is_current,
    hard_region_cache_is_current,
    load_detection_cache,
    load_hard_region_cache,
    save_detection_cache,
    save_hard_region_cache,
)


def _cache(metadata: dict) -> DetectionCache:
    return DetectionCache(
        image_path="synthetic.jpg",
        image_shape=(32, 32),
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        classes=np.zeros((0,), dtype=np.float32),
        feature=np.zeros((4,), dtype=np.float32),
        feature_layers=(10,),
        objectness_map=np.zeros((1, 16, 16), dtype=np.float32),
        spatial_feature_map=np.zeros((4, 16, 16), dtype=np.float32),
        metadata=metadata,
    )


class DetectionCacheMetadataTest(unittest.TestCase):
    def test_expected_metadata_mismatch_invalidates_cache(self) -> None:
        metadata = {"imgsz": 640, "feature_layers": (10,), "weights": {"path": "model.pt"}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "det.npz"
            save_detection_cache(path, _cache(metadata))

            self.assertTrue(detection_cache_is_current(path, metadata))
            self.assertFalse(detection_cache_is_current(path, {**metadata, "imgsz": 320}))

    def test_load_round_trips_metadata(self) -> None:
        metadata = {"imgsz": 640, "conf": 0.01}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "det.npz"
            save_detection_cache(path, _cache(metadata))

            loaded = load_detection_cache(path)

        self.assertEqual(loaded.metadata, metadata)

    def test_hard_region_metadata_invalidates_stale_cache(self) -> None:
        metadata = {"target_classes": (0, 2), "small_area_ratio": 0.0004}
        cache = HardRegionCache(
            image_path="synthetic.jpg",
            image_shape=(32, 32),
            hard_boxes=np.zeros((0, 4), dtype=np.float32),
            small_gt_boxes=np.zeros((0, 4), dtype=np.float32),
            gt_boxes=np.zeros((0, 4), dtype=np.float32),
            matched_iou=np.zeros((0,), dtype=np.float32),
            matched_score=np.zeros((0,), dtype=np.float32),
            metadata=metadata,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hard.npz"
            save_hard_region_cache(path, cache)
            self.assertTrue(hard_region_cache_is_current(path, metadata))
            self.assertFalse(
                hard_region_cache_is_current(
                    path,
                    {"target_classes": (0, 3), "small_area_ratio": 0.0004},
                )
            )
            loaded = load_hard_region_cache(path)
        self.assertEqual(loaded.metadata, {"small_area_ratio": 0.0004, "target_classes": [0, 2]})


if __name__ == "__main__":
    unittest.main()
