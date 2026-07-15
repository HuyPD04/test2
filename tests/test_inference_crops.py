from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.inference.crops import run_yolo_on_crops


class _EmptyResult:
    boxes = None
    speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}


class _Model:
    def predict(self, crops, **_kwargs):
        return [_EmptyResult() for _ in crops]


class InferenceCropTest(unittest.TestCase):
    def test_source_image_avoids_reloading_the_same_file(self) -> None:
        source = np.zeros((100, 120, 3), dtype=np.uint8)
        rois = [
            np.asarray([0, 0, 40, 40], dtype=np.float32),
            np.asarray([40, 40, 80, 80], dtype=np.float32),
        ]
        with patch("rl_sahi.inference.crops.cv2.imread") as imread:
            outputs = run_yolo_on_crops(
                _Model(),
                [Path("image.jpg"), Path("image.jpg")],
                rois,
                imgsz=640,
                conf=0.3,
                iou=0.7,
                max_det=3000,
                device="cpu",
                source_image=source,
            )

        imread.assert_not_called()
        self.assertEqual(len(outputs), 2)


if __name__ == "__main__":
    unittest.main()
