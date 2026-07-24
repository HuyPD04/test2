from __future__ import annotations

import numpy as np

from anchor_zoom_rl.config import RewardConfig
from anchor_zoom_rl.core.reward import crop_step_outcome, match_stats
from anchor_zoom_rl.core.types import Detections


def test_reward_values_new_true_positive_and_crop_cost() -> None:
    ground_truth = Detections(
        np.asarray([[10, 10, 20, 20]], dtype=np.float32),
        np.ones((1,), dtype=np.float32),
        np.zeros((1,), dtype=np.int64),
    )
    before_predictions = Detections.empty()
    after_predictions = Detections(
        np.asarray([[10, 10, 20, 20]], dtype=np.float32),
        np.asarray([0.9], dtype=np.float32),
        np.zeros((1,), dtype=np.int64),
    )
    cfg = RewardConfig(
        utility_weight=1.0,
        tp_weight=2.0,
        small_tp_weight=0.0,
        fp_weight=1.0,
        crop_cost=0.5,
        overlap_penalty=0.0,
        min_utility=0.1,
    )
    before = match_stats(before_predictions, ground_truth, (100, 100), 0.5, 0.001)
    after = match_stats(after_predictions, ground_truth, (100, 100), 0.5, 0.001)
    outcome = crop_step_outcome(before, after, 0.9, 0.0, 1, cfg)

    assert outcome.accepted
    assert outcome.tp_gain == 1
    assert outcome.reward == 2.4


def test_empty_crop_is_rejected() -> None:
    cfg = RewardConfig(crop_cost=0.3, empty_penalty=0.7)
    empty_stats = match_stats(
        Detections.empty(), Detections.empty(), (100, 100), 0.5, 0.001
    )
    outcome = crop_step_outcome(empty_stats, empty_stats, 0.0, 0.0, 0, cfg)
    assert not outcome.accepted
    assert outcome.reason == "empty"
    assert outcome.reward == -1.0


def test_low_reliability_crop_is_rejected() -> None:
    cfg = RewardConfig(
        crop_cost=0.3,
        rejected_penalty=0.5,
        overlap_penalty=0.0,
        min_reliability=0.35,
    )
    empty_stats = match_stats(
        Detections.empty(), Detections.empty(), (100, 100), 0.5, 0.001
    )
    outcome = crop_step_outcome(
        empty_stats,
        empty_stats,
        utility=0.8,
        overlap=0.0,
        num_crop_detections=3,
        cfg=cfg,
        reliability=0.2,
    )
    assert not outcome.accepted
    assert outcome.reason == "low_reliability"
