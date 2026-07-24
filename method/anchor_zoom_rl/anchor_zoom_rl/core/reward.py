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


@dataclass(slots=True)
class HardActionSupervision:
    targets: np.ndarray
    target_mask: np.ndarray
    newly_covered_counts: np.ndarray
    reachable_hard_count: int
    reachable_regular_count: int


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
    newly_covered_hard: int = 0,
) -> StepOutcome:
    coverage_bonus = min(
        cfg.hard_coverage_weight * max(int(newly_covered_hard), 0),
        cfg.hard_coverage_max_bonus,
    )
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
        reward = (
            -cfg.crop_cost
            - penalty
            - cfg.overlap_penalty * overlap
            + coverage_bonus
        )
        return StepOutcome(
            accepted=False,
            reward=float(reward),
            utility=0.0,
            tp_gain=0,
            hard_tp_gain=0,
            fp_gain=0,
            small_tp_gain=0,
            hard_coverage_gain=int(newly_covered_hard),
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
        + coverage_bonus
    )
    return StepOutcome(
        accepted=True,
        reward=float(reward),
        utility=float(utility),
        tp_gain=int(tp_gain),
        hard_tp_gain=int(hard_tp_gain),
        fp_gain=int(fp_gain),
        small_tp_gain=int(small_tp_gain),
        hard_coverage_gain=int(newly_covered_hard),
        reason="accepted",
    )


def build_hard_action_supervision(
    ground_truth: Detections,
    matched_gt: np.ndarray,
    hard_mask: np.ndarray,
    action_rois: np.ndarray,
    action_mask: np.ndarray,
    attempted_rois: list[np.ndarray],
) -> HardActionSupervision:
    rois = np.asarray(action_rois, dtype=np.float32).reshape(-1, 4)
    valid = np.asarray(action_mask, dtype=bool).reshape(-1)
    if len(rois) != len(valid):
        raise ValueError("Action ROIs and action mask must have the same length")
    matched = np.asarray(matched_gt, dtype=bool).reshape(-1)
    hard = np.asarray(hard_mask, dtype=bool).reshape(-1)
    if len(matched) != len(ground_truth) or len(hard) != len(ground_truth):
        raise ValueError("Matched and hard masks must align with ground truth")

    targets = np.zeros((len(rois),), dtype=np.float32)
    newly_covered = np.zeros((len(rois),), dtype=np.int32)
    if len(ground_truth) == 0 or not valid.any():
        return HardActionSupervision(targets, valid, newly_covered, 0, 0)

    centers = (ground_truth.boxes[:, :2] + ground_truth.boxes[:, 2:]) * 0.5
    inside = _centers_inside_rois(centers, rois)
    inside[:, ~valid] = False
    unmatched = ~matched
    unmatched_hard = unmatched & hard
    unmatched_regular = unmatched & ~hard
    targets = (inside & unmatched_hard[:, None]).any(axis=0).astype(np.float32)

    previously_covered = np.zeros((len(ground_truth),), dtype=bool)
    if attempted_rois:
        attempted = np.asarray(attempted_rois, dtype=np.float32).reshape(-1, 4)
        previously_covered = _centers_inside_rois(centers, attempted).any(axis=1)
    new_hard = unmatched_hard & ~previously_covered
    newly_covered = (inside & new_hard[:, None]).sum(axis=0).astype(np.int32)
    reachable = inside.any(axis=1)
    return HardActionSupervision(
        targets=targets,
        target_mask=valid,
        newly_covered_counts=newly_covered,
        reachable_hard_count=int((reachable & unmatched_hard).sum()),
        reachable_regular_count=int((reachable & unmatched_regular).sum()),
    )


def stop_reward(
    reachable_hard_count: int,
    reachable_regular_count: int,
    cfg: RewardConfig,
) -> float:
    hard_count = max(int(reachable_hard_count), 0)
    if hard_count:
        penalty = cfg.stop_hard_base_penalty + cfg.stop_hard_per_gt * hard_count
        return -float(min(penalty, cfg.stop_hard_max_penalty))
    if int(reachable_regular_count) > 0:
        return -float(cfg.stop_early_penalty)
    return float(cfg.stop_bonus)


def covered_ground_truth_mask(
    ground_truth: Detections,
    rois: list[np.ndarray],
) -> np.ndarray:
    if len(ground_truth) == 0 or not rois:
        return np.zeros((len(ground_truth),), dtype=bool)
    centers = (ground_truth.boxes[:, :2] + ground_truth.boxes[:, 2:]) * 0.5
    values = np.asarray(rois, dtype=np.float32).reshape(-1, 4)
    return _centers_inside_rois(centers, values).any(axis=1)


def _centers_inside_rois(centers: np.ndarray, rois: np.ndarray) -> np.ndarray:
    return (
        (centers[:, None, 0] >= rois[None, :, 0])
        & (centers[:, None, 0] <= rois[None, :, 2])
        & (centers[:, None, 1] >= rois[None, :, 1])
        & (centers[:, None, 1] <= rois[None, :, 3])
    )
