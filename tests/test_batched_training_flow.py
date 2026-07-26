from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.actions import Action
from rl_sahi.common.cache import DetectionCache, HardRegionCache
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.pipeline import get_initial_detection
from rl_sahi.rl.batched_trainer import (
    TrainConfig,
    TransitionRecord,
    _push_n_step_slice,
    batched_train_dqn,
)
from rl_sahi.rl.crop_outcome import CropOutcome, CropOutcomeEvaluator
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.replay import ReplayBuffer
from rl_sahi.rl.state_config import StateConfig


def _empty_detection(image_path: str = "image.jpg") -> DetectionCache:
    return DetectionCache(
        image_path=image_path,
        image_shape=(32, 32),
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        classes=np.zeros((0,), dtype=np.float32),
        feature=np.zeros((4,), dtype=np.float32),
        feature_layers=(16,),
        objectness_map=np.zeros((1, 4, 4), dtype=np.float32),
        spatial_feature_map=np.zeros((4, 4, 4), dtype=np.float32),
    )


class BatchedTrainingFlowTest(unittest.TestCase):
    def test_trainer_collects_all_workers_before_one_crop_batch(self) -> None:
        detection = _empty_detection()
        hard = HardRegionCache(
            image_path=detection.image_path,
            image_shape=detection.image_shape,
            hard_boxes=np.array([[14, 14, 18, 18]], dtype=np.float32),
            small_gt_boxes=np.array([[14, 14, 18, 18]], dtype=np.float32),
            gt_boxes=np.array([[14, 14, 18, 18]], dtype=np.float32),
            matched_iou=np.zeros((1,), dtype=np.float32),
            matched_score=np.zeros((1,), dtype=np.float32),
        )

        class FakeDataset:
            def __init__(self, **_kwargs) -> None:
                pass

            def first_detection(self):
                return detection

            def random_episode(self):
                return detection, hard

            def __len__(self) -> int:
                return 2

        class FakeEvaluator:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def full_predictions(self, _det):
                return (
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                )

            def initial_new_count(self, *_args) -> int:
                return 0

            def should_skip_terminal(self, _info) -> bool:
                return False

            def crop_predictions_many(self, paths, rois):
                self.batch_sizes.append(len(rois))
                empty = (
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                )
                return [empty for _ in paths]

            def evaluate_from_predictions(self, **_kwargs):
                return CropOutcome(
                    boxes=np.zeros((0, 4), dtype=np.float32),
                    scores=np.zeros((0,), dtype=np.float32),
                    classes=np.zeros((0,), dtype=np.float32),
                    new_detection_gain=1,
                    new_detection_utility=1.0,
                    new_detection_max_score=1.0,
                    accepted_new_count_after=1,
                    tp_gain=1,
                    fp_gain=0,
                    reward=1.0,
                    accepted=True,
                )

        evaluator = FakeEvaluator()
        train_cfg = TrainConfig(
            episodes=2,
            num_envs=2,
            batch_size=2,
            min_replay=999,
            hidden_dim=8,
            use_spatial_cnn=False,
            epsilon_start=0.0,
            epsilon_end=0.0,
            guide_prob_start=0.0,
            guide_prob_end=0.0,
            val_split="",
            eval_benchmark_images=0,
            resume=False,
            use_curriculum=False,
            preload_cache=False,
        )
        env_cfg = EnvConfig(
            max_steps=1,
            max_slices=2,
            use_gpu_box_ops=False,
            use_action_mask=False,
        )
        infer_cfg = InferenceConfig(
            max_slice_attempts=2,
            crop_batch_size=8,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "rl_sahi.rl.batched_trainer.CachedEpisodeDataset",
                    FakeDataset,
                ),
                patch("rl_sahi.rl.batched_trainer.load_yolo", return_value=object()),
                patch(
                    "rl_sahi.rl.batched_trainer.make_crop_outcome_evaluator",
                    return_value=evaluator,
                ),
            ):
                checkpoint = batched_train_dqn(
                    image_root=Path(directory),
                    cache_root=Path(directory),
                    split="train",
                    out_dir=Path(directory) / "out",
                    cfg=train_cfg,
                    env_cfg=env_cfg,
                    state_cfg=StateConfig(grid_size=4, spatial_feature_channels=4),
                    device_name="cpu",
                    eval_weights=Path(directory) / "weights.pt",
                    infer_cfg=infer_cfg,
                )

            self.assertEqual(evaluator.batch_sizes, [4])
            self.assertTrue(checkpoint.exists())

    def test_crop_outcome_uses_configured_micro_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = InferenceConfig(crop_batch_size=3, device="cpu")
            evaluator = CropOutcomeEvaluator(
                model=object(),
                image_root=Path(directory),
                label_root=None,
                cache_root=Path(directory),
                split="train",
                infer_cfg=cfg,
                use_cache=False,
            )
            batch_sizes: list[int] = []

            def fake_predict(_model, image_paths, rois, **_kwargs):
                batch_sizes.append(len(rois))
                empty = (
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                )
                return [empty for _ in image_paths]

            paths = [Path(directory) / f"{index}.jpg" for index in range(7)]
            rois = [np.array([0, 0, 16, 16], dtype=np.float32) for _ in paths]
            with patch("rl_sahi.rl.crop_outcome.run_yolo_on_crops", side_effect=fake_predict):
                outputs = evaluator.crop_predictions_many(paths, rois)

            self.assertEqual(batch_sizes, [3, 3, 1])
            self.assertEqual(len(outputs), 7)

    def test_terminal_reward_is_applied_before_n_step_replay(self) -> None:
        state = np.zeros((1,), dtype=np.float32)
        valid = np.ones((len(Action),), dtype=bool)
        transitions = [
            TransitionRecord(state, Action.RIGHT, 1.0, state, False, valid),
            TransitionRecord(state, Action.DOWN, 2.0, state, False, valid),
            TransitionRecord(state, Action.STOP, 0.0, state, True, valid),
        ]
        transitions[-1].reward = 10.0
        replay = ReplayBuffer(8)

        _push_n_step_slice(transitions, replay, n_step=3, gamma=0.5)

        rewards = [item[2] for item in replay.buffer]
        self.assertEqual(rewards, [4.5, 7.0, 10.0])

    def test_infer_without_cache_neither_loads_nor_writes_disk_cache(self) -> None:
        detection = _empty_detection()
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.jpg"
            with (
                patch(
                    "rl_sahi.inference.pipeline.detect_one_image",
                    return_value=detection,
                ) as detect_mock,
                patch(
                    "rl_sahi.inference.pipeline.detection_cache_is_current"
                ) as current_mock,
                patch("rl_sahi.inference.pipeline.save_detection_cache") as save_mock,
            ):
                result = get_initial_detection(
                    model=object(),
                    weights=None,
                    image_path=image_path,
                    weights_imgsz=640,
                    full_conf=0.01,
                    full_iou=0.7,
                    max_det=3000,
                    device="cpu",
                    feature_layers=(16,),
                    aux_grid_size=4,
                    spatial_feature_channels=4,
                    cache_root=Path(directory) / "cache",
                    split="val",
                    use_cache=False,
                )

            self.assertIs(result, detection)
            detect_mock.assert_called_once()
            current_mock.assert_not_called()
            save_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
