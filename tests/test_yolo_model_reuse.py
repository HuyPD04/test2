from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.detection.yolo import load_yolo_variants


class YoloModelReuseTest(unittest.TestCase):
    def test_same_weights_path_loads_one_shared_model(self) -> None:
        shared_model = object()
        with patch("rl_sahi.detection.yolo.load_yolo", return_value=shared_model) as load:
            state_model, full_model, crop_model = load_yolo_variants(
                Path("best.pt"),
                device="cpu",
                full_weights=Path(".") / "best.pt",
                crop_weights=Path("best.pt"),
            )

        self.assertIs(state_model, shared_model)
        self.assertIs(full_model, shared_model)
        self.assertIs(crop_model, shared_model)
        self.assertEqual(load.call_count, 1)

    def test_distinct_weights_paths_load_distinct_models(self) -> None:
        models = [object(), object(), object()]
        with patch("rl_sahi.detection.yolo.load_yolo", side_effect=models) as load:
            state_model, full_model, crop_model = load_yolo_variants(
                Path("state.pt"),
                device="cpu",
                full_weights=Path("full.pt"),
                crop_weights=Path("crop.pt"),
            )

        self.assertIs(state_model, models[0])
        self.assertIs(full_model, models[1])
        self.assertIs(crop_model, models[2])
        self.assertEqual(load.call_count, 3)


if __name__ == "__main__":
    unittest.main()
