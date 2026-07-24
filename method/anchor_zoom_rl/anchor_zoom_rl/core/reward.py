from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import RewardConfig
from .geometry import box_area, iou_matrix
from .types import Detections, StepOutcome


@dataclass(slots=True)
class MatchStats:
    true_positives: int
    false_positives: int
    hard_true_positives: int
    small_true_positives: int
    matched_gt: np.ndarray


def match_stats(
    predictions: Detections,
    ground_truth: Detections,
    image_shape: tuple[int, int],
    iou_threshold: float,
    small_area_ratio: float,
    hard_gt_mask: np.ndarray | None = None,
) -> MatchStats:
    if hard_gt_mask is None:
        hard_gt_mask = np.zeros((len(ground_truth),), dtype=bool)
    else:
        hard_gt_mask = np.asarray(hard_gt_mask, dtype=bool).reshape(-1)
        if len(hard_gt_mask) != len(ground_truth):
            raise ValueError("hard_gt_mask must align with ground_truth")
    if len(predictions) == 0:
        return MatchStats(
            0,
            0,
            0,
            0,
            np.zeros((len(ground_truth),), dtype=bool),
        )
    matched_gt = np.zeros((len(ground_truth),), dtype=bool)
    true_positives = 0
    hard_true_positives = 0
    small_true_positives = 0
    image_area = float(max(image_shape[0] * image_shape[1], 1))
    gt_small = box_area(ground_truth.boxes) / image_area <= float(small_area_ratio)
    order = np.argsort(-predictions.scores)
    for pred_index in order:
        candidates = np.flatnonzero(
            (ground_truth.classes == predictions.classes[pred_index]) & ~matched_gt
        )
        if len(candidates) == 0:
            continue
        overlaps = iou_matrix(
            predictions.boxes[[pred_index]], ground_truth.boxes[candidates]
        )[0]
        best_local = int(np.argmax(overlaps))
        if float(overlaps[best_local]) >= float(iou_threshold):
            gt_index = int(candidates[best_local])
            matched_gt[gt_index] = True
            true_positives += 1
            hard_true_positives += int(hard_gt_mask[gt_index])
            small_true_positives += int(gt_small[gt_index])
    return MatchStats(
        true_positives=true_positives,
        false_positives=len(predictions) - true_positives,
        hard_true_positives=hard_true_positives,
        small_true_positives=small_true_positives,
        matched_gt=matched_gt,
    )


def crop_step_outcome(
    before: MatchStats,
    after: MatchStats,
    utility: float,
    overlap: float,
    num_crop_detections: int,
    cfg: RewardConfig,
    reliability: float = 1.0,
) -> StepOutcome:
    accepted = (
        num_crop_detections >= cfg.min_crop_detections
        and utility >= cfg.min_utility
        and reliability >= cfg.min_reliability
    )
    if not accepted:
        if num_crop_detections == 0:
            reason = "empty"
        elif utility < cfg.min_utility:
            reason = "no_gain"
        else:
            reason = "low_reliability"
        penalty = cfg.empty_penalty if num_crop_detections == 0 else cfg.rejected_penalty
        reward = -cfg.crop_cost - penalty - cfg.overlap_penalty * overlap
        return StepOutcome(
            accepted=False,
            reward=float(reward),
            utility=0.0,
            tp_gain=0,
            hard_tp_gain=0,
            fp_gain=0,
            small_tp_gain=0,
            reason=reason,
        )

    tp_gain = after.true_positives - before.true_positives
    hard_tp_gain = after.hard_true_positives - before.hard_true_positives
    fp_gain = after.false_positives - before.false_positives
    small_tp_gain = after.small_true_positives - before.small_true_positives
    reward = (
        cfg.utility_weight * utility
        + cfg.tp_weight * tp_gain
        + cfg.hard_tp_weight * hard_tp_gain
        + cfg.small_tp_weight * small_tp_gain
        - cfg.fp_weight * max(fp_gain, 0)
        - cfg.crop_cost
        - cfg.overlap_penalty * overlap
    )
    return StepOutcome(
        accepted=True,
        reward=float(reward),
        utility=float(utility),
        tp_gain=int(tp_gain),
        hard_tp_gain=int(hard_tp_gain),
        fp_gain=int(fp_gain),
        small_tp_gain=int(small_tp_gain),
        reason="accepted",
    )
