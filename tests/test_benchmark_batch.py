from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.cache import DetectionCache
from rl_sahi.eval.benchmark import _predict_rl_sahi
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.state_config import StateConfig


def _empty_prediction() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )


class BenchmarkBatchInferenceTest(unittest.TestCase):
    def test_batched_rl_sahi_predicts_all_candidates_in_one_call(self) -> None:
        det = DetectionCache(
            image_path="image.jpg",
            image_shape=(100, 100),
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            classes=np.zeros((0,), dtype=np.float32),
            feature=np.zeros((1,), dtype=np.float32),
            feature_layers=(16,),
            objectness_map=np.zeros((16, 16), dtype=np.float32),
            spatial_feature_map=np.zeros((4, 16, 16), dtype=np.float32),
        )
        rois = [
            np.asarray([0, 0, 20, 20], dtype=np.float32),
            np.asarray([30, 0, 50, 20], dtype=np.float32),
            np.asarray([60, 0, 80, 20], dtype=np.float32),
        ]
        rollout_results = [(roi, ["stop"], {}) for roi in rois]

        with (
            patch("rl_sahi.eval.benchmark.SliceEnv", return_value=object()),
            patch("rl_sahi.eval.benchmark.rollout_one_slice", side_effect=rollout_results),
            patch(
                "rl_sahi.eval.benchmark.run_yolo_on_crops",
                return_value=[_empty_prediction() for _ in rois],
            ) as predict_crops,
        ):
            model = object()
            _boxes, _scores, _classes, accepted, inferred = _predict_rl_sahi(
                model=model,
                full_model=model,
                crop_model=object(),
                policy=object(),
                device_t=object(),
                image_path=Path("image.jpg"),
                det=det,
                cfg=InferenceConfig(
                    batched_inference=True,
                    max_slice_attempts=3,
                    target_classes=(),
                ),
                env_cfg=EnvConfig(max_slices=3),
                state_cfg=StateConfig(),
            )

        self.assertEqual(predict_crops.call_count, 1)
        self.assertEqual(len(predict_crops.call_args.args[1]), 3)
        self.assertEqual(len(predict_crops.call_args.args[2]), 3)
        self.assertEqual(inferred, 3)
        self.assertEqual(accepted, 0)

    def test_batched_rl_sahi_filters_crop_boundary_boxes(self) -> None:
        det = DetectionCache(
            image_path="image.jpg",
            image_shape=(100, 100),
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            classes=np.zeros((0,), dtype=np.float32),
            feature=np.zeros((1,), dtype=np.float32),
            feature_layers=(16,),
            objectness_map=np.zeros((16, 16), dtype=np.float32),
            spatial_feature_map=np.zeros((4, 16, 16), dtype=np.float32),
        )
        roi = np.asarray([10, 10, 50, 50], dtype=np.float32)
        boundary_prediction = (
            np.asarray([[10, 20, 25, 35]], dtype=np.float32),
            np.asarray([0.9], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
        )

        with (
            patch("rl_sahi.eval.benchmark.SliceEnv", return_value=object()),
            patch("rl_sahi.eval.benchmark.rollout_one_slice", return_value=(roi, ["stop"], {})),
            patch("rl_sahi.eval.benchmark.run_yolo_on_crops", return_value=[boundary_prediction]),
        ):
            model = object()
            boxes, _scores, _classes, accepted, inferred = _predict_rl_sahi(
                model=model,
                full_model=model,
                crop_model=object(),
                policy=object(),
                device_t=object(),
                image_path=Path("image.jpg"),
                det=det,
                cfg=InferenceConfig(
                    batched_inference=True,
                    max_slice_attempts=1,
                    target_classes=(),
                    min_new_detection_score=0.1,
                ),
                env_cfg=EnvConfig(max_slices=1),
                state_cfg=StateConfig(),
            )

        self.assertEqual(inferred, 1)
        self.assertEqual(accepted, 0)
        self.assertEqual(len(boxes), 0)


if __name__ == "__main__":
    unittest.main()
