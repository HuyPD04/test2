from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.detection.features import SpatialFeatureCollector


class _FakeYolo:
    def __init__(self, module_count: int = 7) -> None:
        self.model = type("FakeModel", (), {})()
        self.model.model = nn.ModuleList([nn.Identity() for _ in range(module_count)])


class SpatialFeatureCollectorTest(unittest.TestCase):
    def test_collects_fixed_width_maps_from_layer_six(self) -> None:
        yolo = _FakeYolo()
        activation = torch.arange(8 * 4 * 4, dtype=torch.float32).reshape(1, 8, 4, 4)

        with SpatialFeatureCollector(yolo, (6,)) as collector:
            collector.clear()
            yolo.model.model[6](activation)
            maps = collector.maps(grid_size=2, spatial_feature_channels=4)

        self.assertEqual(maps.shape, (4, 2, 2))
        self.assertEqual(maps.dtype, np.float32)
        self.assertTrue(np.isfinite(maps).all())
        self.assertGreater(float(np.abs(maps).sum()), 0.0)

    def test_rejects_out_of_range_layer(self) -> None:
        with self.assertRaisesRegex(ValueError, "Spatial feature layer 7 is out of range"):
            with SpatialFeatureCollector(_FakeYolo(), (7,)):
                pass


if __name__ == "__main__":
    unittest.main()
