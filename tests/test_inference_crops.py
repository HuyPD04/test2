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


class _FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Boxes:
    def __init__(self):
        self.xyxy = _FakeTensor([[1.25, 2.5, 11.75, 22.25]])
        self.conf = _FakeTensor([0.9])
        self.cls = _FakeTensor([3.0])

    def __len__(self):
        return 1


class _BoxResult:
    boxes = _Boxes()
    speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}


class _Model:
    def predict(self, crops, **_kwargs):
        return [_EmptyResult() for _ in crops]


class _BoxModel:
    def predict(self, crops, **_kwargs):
        return [_BoxResult() for _ in crops]


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

    def test_crop_boxes_are_mapped_back_with_rounded_clamped_offset(self) -> None:
        source = np.zeros((100, 120, 3), dtype=np.uint8)
        roi = np.asarray([10.2, 20.6, 50.7, 80.3], dtype=np.float32)

        boxes, scores, classes = run_yolo_on_crops(
            _BoxModel(),
            [Path("image.jpg")],
            [roi],
            imgsz=512,
            conf=0.05,
            iou=0.7,
            max_det=3000,
            device="cpu",
            source_image=source,
        )[0]

        np.testing.assert_allclose(
            boxes,
            np.asarray([[11.25, 23.5, 21.75, 43.25]], dtype=np.float32),
        )
        np.testing.assert_allclose(scores, np.asarray([0.9], dtype=np.float32))
        np.testing.assert_allclose(classes, np.asarray([3.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
