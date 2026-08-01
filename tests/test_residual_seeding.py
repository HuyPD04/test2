from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.boxes import centers
from rl_sahi.common.cache import DetectionCache
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig


def _detection(
    boxes: np.ndarray,
    scores: np.ndarray,
    objectness: np.ndarray,
) -> DetectionCache:
    return DetectionCache(
        image_path="synthetic.jpg",
        image_shape=(160, 160),
        boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        scores=np.asarray(scores, dtype=np.float32).reshape(-1),
        classes=np.zeros((len(scores),), dtype=np.float32),
        feature=np.zeros((4,), dtype=np.float32),
        feature_layers=(16,),
        objectness_map=np.asarray(objectness, dtype=np.float32).reshape(1, 16, 16),
        spatial_feature_map=np.zeros((4, 16, 16), dtype=np.float32),
    )


class ResidualSeedingTest(unittest.TestCase):
    def test_residual_heatmap_avoids_high_confidence_peak(self) -> None:
        objectness = np.zeros((16, 16), dtype=np.float32)
        objectness[2, 2] = 1.0
        objectness[12, 12] = 0.8
        detection = _detection(
            np.asarray([[15, 15, 35, 35], [120, 120, 130, 130]]),
            np.asarray([0.9, 0.25]),
            objectness,
        )
        env = SliceEnv(
            detection,
            None,
            env_cfg=EnvConfig(
                residual_high_conf_penalty=2.0,
                seed_nms_radius=2,
            ),
            state_cfg=StateConfig(grid_size=16),
        )

        target, _score = env._heatmap_targets(
            top_k=1,
            include_previous=False,
        )[0]

        self.assertGreater(float(target[0]), 80.0)
        self.assertGreater(float(target[1]), 80.0)

    def test_ranked_seed_uses_spatially_distinct_peak(self) -> None:
        objectness = np.zeros((16, 16), dtype=np.float32)
        objectness[2, 2] = 1.0
        objectness[2, 3] = 0.95
        objectness[12, 12] = 0.8
        detection = _detection(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            objectness,
        )
        cfg = EnvConfig(
            residual_objectness_weight=1.0,
            residual_proposal_weight=0.0,
            residual_small_weight=0.0,
            residual_high_conf_penalty=0.0,
            seed_topk=2,
            seed_nms_radius=2,
        )
        first = SliceEnv(
            detection,
            None,
            env_cfg=cfg,
            state_cfg=StateConfig(grid_size=16),
            seed_rank=0,
        )
        second = SliceEnv(
            detection,
            None,
            env_cfg=cfg,
            state_cfg=StateConfig(grid_size=16),
            seed_rank=1,
        )

        first_center = centers(first.roi.reshape(1, 4))[0]
        second_center = centers(second.roi.reshape(1, 4))[0]

        self.assertGreater(
            float(np.linalg.norm(first_center - second_center)),
            50.0,
        )

    def test_cached_ranked_seeds_match_per_attempt_initialization(self) -> None:
        objectness = np.zeros((16, 16), dtype=np.float32)
        objectness[2, 2] = 1.0
        objectness[12, 12] = 0.8
        detection = _detection(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            objectness,
        )
        cfg = EnvConfig(
            residual_objectness_weight=1.0,
            residual_proposal_weight=0.0,
            residual_small_weight=0.0,
            residual_high_conf_penalty=0.0,
            seed_topk=2,
            seed_nms_radius=2,
        )
        state_cfg = StateConfig(grid_size=16)
        class_mapping = ClassMapping()
        static = SliceEnv.build_static_context(
            detection,
            state_cfg,
            (),
            class_mapping,
        )
        cached_targets = SliceEnv.precompute_seed_targets(
            detection,
            cfg,
            state_cfg,
            (),
            class_mapping,
            static,
            max_attempts=3,
        )

        self.assertEqual(len(cached_targets), 2)
        for rank in range(3):
            eager = SliceEnv(
                detection,
                None,
                env_cfg=cfg,
                state_cfg=state_cfg,
                static_context=static,
                seed_rank=rank,
                lazy_reset=True,
            )
            cached = SliceEnv(
                detection,
                None,
                env_cfg=cfg,
                state_cfg=state_cfg,
                static_context=static,
                seed_rank=rank,
                seed_target=(cached_targets[rank] if rank < len(cached_targets) else None),
                lazy_reset=True,
            )
            np.testing.assert_array_equal(eager.reset(), cached.reset())
            np.testing.assert_array_equal(eager.roi, cached.roi)
            np.testing.assert_array_equal(
                eager.policy_action_mask(),
                cached.policy_action_mask(),
            )

    def test_residual_heatmap_excludes_previous_roi(self) -> None:
        objectness = np.zeros((16, 16), dtype=np.float32)
        objectness[2, 2] = 1.0
        objectness[12, 12] = 0.8
        detection = _detection(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            objectness,
        )
        env = SliceEnv(
            detection,
            None,
            env_cfg=EnvConfig(
                residual_objectness_weight=1.0,
                residual_proposal_weight=0.0,
                residual_small_weight=0.0,
                residual_high_conf_penalty=0.0,
            ),
            state_cfg=StateConfig(grid_size=16),
            previous_rois=np.asarray([[0, 0, 45, 45]], dtype=np.float32),
        )

        target, _score = env._heatmap_targets(
            top_k=1,
            include_previous=True,
        )[0]

        self.assertGreater(float(target[0]), 80.0)
        self.assertGreater(float(target[1]), 80.0)


if __name__ == "__main__":
    unittest.main()
