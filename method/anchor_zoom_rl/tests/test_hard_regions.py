from __future__ import annotations

import numpy as np

from anchor_zoom_rl.config import RewardConfig
from anchor_zoom_rl.core.hard_regions import build_hard_regions
from anchor_zoom_rl.core.reward import (
    build_hard_action_supervision,
    crop_step_outcome,
    match_stats,
    stop_reward,
)
from anchor_zoom_rl.core.types import Detections


def test_hard_regions_include_missed_and_low_confidence_ground_truth() -> None:
    ground_truth = Detections(
        boxes=np.asarray(
            [
                [10, 10, 20, 20],
                [30, 30, 40, 40],
                [50, 50, 60, 60],
            ],
            dtype=np.float32,
        ),
        scores=np.ones((3,), dtype=np.float32),
        classes=np.asarray([0, 0, 1], dtype=np.int64),
    )
    full = Detections(
        boxes=np.asarray(
            [
                [10, 10, 20, 20],
                [30, 30, 40, 40],
            ],
            dtype=np.float32,
        ),
        scores=np.asarray([0.9, 0.1], dtype=np.float32),
        classes=np.asarray([0, 0], dtype=np.int64),
    )
    regions = build_hard_regions(
        full,
        ground_truth,
        match_iou=0.5,
        low_confidence=0.25,
    )

    assert regions.hard_mask.tolist() == [False, True, True]
    assert np.allclose(regions.best_score, [0.9, 0.1, 0.0])
    assert len(regions.hard_boxes) == 2


def test_reward_adds_bonus_when_crop_recovers_hard_gt() -> None:
    ground_truth = Detections(
        np.asarray([[10, 10, 20, 20]], dtype=np.float32),
        np.ones((1,), dtype=np.float32),
        np.zeros((1,), dtype=np.int64),
    )
    recovered = Detections(
        np.asarray([[10, 10, 20, 20]], dtype=np.float32),
        np.asarray([0.9], dtype=np.float32),
        np.zeros((1,), dtype=np.int64),
    )
    hard_mask = np.asarray([True])
    before = match_stats(
        Detections.empty(),
        ground_truth,
        (100, 100),
        0.5,
        0.001,
        hard_mask,
    )
    after = match_stats(
        recovered,
        ground_truth,
        (100, 100),
        0.5,
        0.001,
        hard_mask,
    )
    cfg = RewardConfig(
        utility_weight=0.0,
        tp_weight=2.0,
        hard_tp_weight=1.5,
        small_tp_weight=0.0,
        crop_cost=0.5,
        overlap_penalty=0.0,
        min_utility=0.1,
    )
    outcome = crop_step_outcome(before, after, 0.9, 0.0, 1, cfg)

    assert outcome.hard_tp_gain == 1
    assert outcome.reward == 3.0


def test_hard_action_supervision_and_dynamic_stop_penalty() -> None:
    ground_truth = Detections(
        np.asarray(
            [[10, 10, 20, 20], [30, 30, 40, 40], [80, 80, 90, 90]],
            dtype=np.float32,
        ),
        np.ones((3,), dtype=np.float32),
        np.zeros((3,), dtype=np.int64),
    )
    supervision = build_hard_action_supervision(
        ground_truth=ground_truth,
        matched_gt=np.asarray([False, False, False]),
        hard_mask=np.asarray([True, True, False]),
        action_rois=np.asarray(
            [[0, 0, 50, 50], [70, 70, 100, 100], [0, 0, 0, 0]],
            dtype=np.float32,
        ),
        action_mask=np.asarray([True, True, False]),
        attempted_rois=[],
    )

    assert supervision.targets.tolist() == [1.0, 0.0, 0.0]
    assert supervision.newly_covered_counts.tolist() == [2, 0, 0]
    assert supervision.reachable_hard_count == 2
    assert supervision.reachable_regular_count == 1
    cfg = RewardConfig(
        stop_hard_base_penalty=1.0,
        stop_hard_per_gt=0.15,
        stop_hard_max_penalty=4.0,
    )
    assert stop_reward(2, 1, cfg) == -1.3
    assert stop_reward(100, 0, cfg) == -4.0
    assert stop_reward(0, 1, cfg) == -cfg.stop_early_penalty
    assert stop_reward(0, 0, cfg) == cfg.stop_bonus
