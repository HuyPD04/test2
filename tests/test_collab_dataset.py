from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.collab.dataset import (
    CollaborativeDatasetConfig,
    build_yolo_crop_dataset,
    project_labels_to_crop,
    roi_to_crop_bounds,
)


class CollaborativeDatasetTest(unittest.TestCase):
    def test_roi_to_crop_bounds_matches_clamped_integer_crop(self) -> None:
        bounds = roi_to_crop_bounds([-2.4, 4.6, 30.4, 120.1], (100, 80))

        self.assertEqual(bounds, (0, 5, 30, 100))

    def test_project_labels_to_crop_clips_and_normalizes_yolo_boxes(self) -> None:
        classes = np.array([0, 1, 2], dtype=np.float32)
        boxes = np.array(
            [
                [10.0, 10.0, 30.0, 30.0],
                [40.0, 40.0, 80.0, 80.0],
                [0.0, 0.0, 5.0, 5.0],
            ],
            dtype=np.float32,
        )

        crop_classes, crop_boxes = project_labels_to_crop(
            classes,
            boxes,
            (20, 20, 60, 60),
            min_visibility=0.25,
            min_box_size=1.0,
        )

        np.testing.assert_array_equal(crop_classes, np.array([0, 1], dtype=np.float32))
        np.testing.assert_allclose(
            crop_boxes,
            np.array(
                [
                    [0.125, 0.125, 0.25, 0.25],
                    [0.75, 0.75, 0.5, 0.5],
                ],
                dtype=np.float32,
            ),
            atol=1e-6,
        )

    def test_build_yolo_crop_dataset_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            label_root = root / "labels"
            metadata_dir = root / "metadata"
            out_dir = root / "collab"
            (image_root / "train").mkdir(parents=True)
            (label_root / "train").mkdir(parents=True)
            metadata_dir.mkdir()

            image_path = image_root / "train" / "sample.jpg"
            image = np.zeros((100, 100, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(image_path), image))
            (label_root / "train" / "sample.txt").write_text(
                "3 0.500000 0.500000 0.400000 0.400000\n",
                encoding="utf-8",
            )
            (metadata_dir / "sample.json").write_text(
                json.dumps(
                    {
                        "image": str(image_path),
                        "slices": [
                            {
                                "attempt_index": 1,
                                "slice_index": 1,
                                "accepted": True,
                                "roi": [20, 20, 80, 80],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = build_yolo_crop_dataset(
                metadata_dir=metadata_dir,
                image_root=image_root,
                label_root=label_root,
                out_dir=out_dir,
                split="train",
                image_paths=[image_path],
                cfg=CollaborativeDatasetConfig(min_visibility=0.5),
            )

            crop_image = out_dir / "images" / "train" / "sample_acc_a001_s001.jpg"
            crop_label = out_dir / "labels" / "train" / "sample_acc_a001_s001.txt"
            self.assertTrue(crop_image.exists())
            self.assertTrue(crop_label.exists())
            self.assertTrue((out_dir / "data.yaml").exists())
            self.assertEqual(summary.crops_written, 1)
            self.assertEqual(summary.labels_written, 1)
            self.assertEqual(crop_label.read_text(encoding="utf-8").strip(), "3 0.500000 0.500000 0.666667 0.666667")


if __name__ == "__main__":
    unittest.main()

