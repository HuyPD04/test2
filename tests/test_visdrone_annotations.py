from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.boxes import covered_area_by_boxes
from rl_sahi.common.data import read_visdrone_det_annotations
from rl_sahi.eval.benchmark import _read_gt


class VisDroneAnnotationTest(unittest.TestCase):
    def test_covered_area_uses_union_of_overlapping_ignore_boxes(self) -> None:
        covered = covered_area_by_boxes(
            np.asarray([[0, 0, 10, 10]], dtype=np.float32),
            np.asarray([[0, 0, 6, 10], [4, 0, 10, 10]], dtype=np.float32),
        )
        np.testing.assert_allclose(covered, np.asarray([100], dtype=np.float32))

    def test_read_visdrone_det_annotations_keeps_ignore_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            annotation_path = Path(tmp) / "sample.txt"
            annotation_path.write_text(
                "\n".join(
                    (
                        "10,20,30,40,1,4,0,0",
                        "50,50,30,30,0,0,0,0",
                        "55,55,10,10,1,1,0,0",
                        "70,10,10,10,1,11,0,0",
                        "0,0,5,5,0,2,0,0",
                    )
                ),
                encoding="utf-8",
            )

            classes, boxes, ignored_boxes = read_visdrone_det_annotations(
                annotation_path,
                image_shape=(100, 100),
            )

        np.testing.assert_array_equal(classes, np.asarray([3], dtype=np.float32))
        np.testing.assert_allclose(
            boxes,
            np.asarray([[10, 20, 40, 60]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            ignored_boxes,
            np.asarray(
                [
                    [50, 50, 80, 80],
                    [70, 10, 80, 20],
                    [0, 0, 5, 5],
                ],
                dtype=np.float32,
            ),
        )

    def test_read_gt_resolves_official_visdrone_layout_from_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            image_dir = image_root / "test"
            annotation_dir = root / "VisDrone2019-DET-test-dev" / "annotations"
            image_dir.mkdir(parents=True)
            annotation_dir.mkdir(parents=True)
            image_path = image_dir / "sample.jpg"
            cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
            (annotation_dir / "sample.txt").write_text(
                "\n".join(
                    (
                        "10,20,30,40,1,4,0,0",
                        "50,50,30,30,0,0,0,0",
                    )
                ),
                encoding="utf-8",
            )

            boxes, classes, ignored_boxes = _read_gt(
                image_path=image_path,
                image_root=image_root,
                label_root=root / "labels",
                target_classes=tuple(range(10)),
                class_mapping=ClassMapping(),
                annotation_root=root,
            )

        np.testing.assert_array_equal(classes, np.asarray([3], dtype=np.float32))
        np.testing.assert_allclose(boxes, np.asarray([[10, 20, 40, 60]], dtype=np.float32))
        np.testing.assert_allclose(ignored_boxes, np.asarray([[50, 50, 80, 80]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
