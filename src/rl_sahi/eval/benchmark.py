from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from rl_sahi.common.boxes import area, box_from_center, centers, clip_boxes, intersection_matrix, iou_matrix
from rl_sahi.common.cache import DetectionCache, file_fingerprint
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.data import image_to_label_path, iter_images, read_yolo_labels
from rl_sahi.common.device import resolve_torch_device
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crops
from rl_sahi.inference.merge import class_aware_nms
from rl_sahi.inference.pipeline import (
    _crop_rejection_reason,
    _attempt_overlap,
    _filter_classes,
    _new_detection_stats,
    _skip_crop_reason,
    get_initial_detection,
)
from rl_sahi.inference.rollout import rollout_one_slice
from rl_sahi.rl.checkpoint import load_policy
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig


@dataclass(slots=True)
class BenchmarkConfig:
    iou_threshold: float = 0.5
    fixed_slice_fraction: float = 0.35
    fixed_overlap: float = 0.2
    budgeted_crop_counts: tuple[int, ...] = (4, 8, 12)
    include_fixed_grid_full: bool = True
    include_gated_variants: bool = True
    proposal_crop_count: int = 8
    proposal_min_conf: float = 0.01
    proposal_max_conf: float = 0.5
    proposal_peak_conf: float = 0.25
    proposal_overlap_threshold: float = 0.7
    small_area_ratio: float | None = 0.0004
    small_area_percentile: float | None = None
    sampling: str = "stratified"
    seed: int = 42
    warmup_images: int = 10
    detector_gflops: float = 21.5
    agent_gflops: float = 0.0
    target_classes: tuple[int, ...] = (0, 2, 3, 5, 8, 9)
    class_mapping: ClassMapping = field(default_factory=ClassMapping)


def _empty_preds() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )


def _read_gt(
    image_path: Path,
    image_root: Path,
    label_root: Path,
    target_classes: tuple[int, ...],
    class_mapping: ClassMapping,
):
    classes, boxes = read_yolo_labels(image_to_label_path(image_path, image_root, label_root), _image_shape(image_path))
    classes = class_mapping.map_label_classes(classes)
    if target_classes:
        mask = np.isin(classes.astype(np.int64), np.asarray(target_classes, dtype=np.int64))
        classes, boxes = classes[mask], boxes[mask]
    return boxes.astype(np.float32), classes.astype(np.float32)


def _image_shape(image_path: Path) -> tuple[int, int]:
    import cv2

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h, w = image.shape[:2]
    return int(h), int(w)


def _small_area_threshold(
    images: list[Path],
    image_root: Path,
    label_root: Path,
    target_classes: tuple[int, ...],
    percentile: float,
    class_mapping: ClassMapping,
) -> float:
    ratios: list[np.ndarray] = []
    for image_path in images:
        boxes, _classes = _read_gt(image_path, image_root, label_root, target_classes, class_mapping)
        if len(boxes) == 0:
            continue
        h, w = _image_shape(image_path)
        ratios.append(area(boxes) / max(float(h * w), 1.0))
    if not ratios:
        return 0.0
    return float(np.percentile(np.concatenate(ratios, axis=0), percentile))


def _resolve_small_area_threshold(
    images: list[Path],
    image_root: Path,
    label_root: Path,
    cfg: BenchmarkConfig,
) -> float:
    if cfg.small_area_ratio is not None:
        return float(cfg.small_area_ratio)
    if cfg.small_area_percentile is None:
        raise ValueError("Set benchmark.small_area_ratio or benchmark.small_area_percentile")
    return _small_area_threshold(
        images,
        image_root,
        label_root,
        cfg.target_classes,
        float(cfg.small_area_percentile),
        cfg.class_mapping,
    )


def select_benchmark_images(
    images: list[Path],
    limit: int | None,
    sampling: str = "stratified",
    seed: int = 42,
) -> list[Path]:
    images = list(images)
    if limit is None or limit >= len(images):
        return images
    if limit <= 0:
        return []
    if sampling == "sequential":
        return images[:limit]
    if sampling != "stratified":
        raise ValueError("benchmark.sampling must be 'stratified' or 'sequential'")
    groups: dict[str, list[Path]] = {}
    for image in images:
        groups.setdefault(image.stem.split("_", 1)[0], []).append(image)
    rng = np.random.default_rng(int(seed))
    group_names = sorted(groups)
    rng.shuffle(group_names)
    for name in group_names:
        rng.shuffle(groups[name])
    selected: list[Path] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for name in group_names:
            if offset < len(groups[name]):
                selected.append(groups[name][offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def _effective_warmup_images(total_images: int, requested: int) -> int:
    if requested < 0:
        raise ValueError("benchmark.warmup_images must be non-negative")
    return min(int(requested), max(int(total_images) - 1, 0))


def _git_revision(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _merge_predictions(
    image_shape: tuple[int, int],
    merge_iou: float,
    boxes_parts: list[np.ndarray],
    scores_parts: list[np.ndarray],
    classes_parts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.concatenate(boxes_parts, axis=0) if boxes_parts else np.zeros((0, 4), dtype=np.float32)
    scores = np.concatenate(scores_parts, axis=0) if scores_parts else np.zeros((0,), dtype=np.float32)
    classes = np.concatenate(classes_parts, axis=0) if classes_parts else np.zeros((0,), dtype=np.float32)
    if len(boxes) == 0:
        return _empty_preds()
    boxes = clip_boxes(boxes, image_shape)
    keep = class_aware_nms(boxes, scores, classes, merge_iou)
    return boxes[keep], scores[keep], classes[keep]


def _full_predictions(det: DetectionCache, cfg: InferenceConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = det.scores >= cfg.output_conf
    boxes, scores = det.boxes[mask], det.scores[mask]
    classes = cfg.class_mapping.map_model_classes(det.classes[mask])
    return _filter_classes(boxes, scores, classes, cfg.target_classes)


def _fixed_grid_rois(image_shape: tuple[int, int], fraction: float, overlap: float) -> list[np.ndarray]:
    h, w = image_shape
    side = max(1.0, min(h, w) * float(fraction))
    stride = max(1.0, side * (1.0 - float(overlap)))
    xs = list(np.arange(0.0, max(w - side, 0.0) + 1.0, stride))
    ys = list(np.arange(0.0, max(h - side, 0.0) + 1.0, stride))
    if not xs or xs[-1] < w - side:
        xs.append(max(w - side, 0.0))
    if not ys or ys[-1] < h - side:
        ys.append(max(h - side, 0.0))
    rois: list[np.ndarray] = []
    for y in ys:
        for x in xs:
            rois.append(np.asarray([x, y, min(x + side, w), min(y + side, h)], dtype=np.float32))
    return rois


def _proposal_quality(scores: np.ndarray, bench_cfg: BenchmarkConfig) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        return scores
    min_conf = float(bench_cfg.proposal_min_conf)
    peak_conf = float(bench_cfg.proposal_peak_conf)
    max_conf = float(bench_cfg.proposal_max_conf)
    up = (scores - min_conf) / max(peak_conf - min_conf, 1e-6)
    down = (max_conf - scores) / max(max_conf - peak_conf, 1e-6)
    return np.clip(np.minimum(up, down), 0.0, 1.0).astype(np.float32)


def _proposal_detections(
    det: DetectionCache,
    cfg: InferenceConfig,
    bench_cfg: BenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(det.scores, dtype=np.float32).reshape(-1)
    mask = (scores >= float(bench_cfg.proposal_min_conf)) & (scores <= float(bench_cfg.proposal_max_conf))
    boxes = np.asarray(det.boxes, dtype=np.float32).reshape(-1, 4)[mask]
    proposal_scores = scores[mask]
    classes = cfg.class_mapping.map_model_classes(det.classes[mask])
    return _filter_classes(boxes, proposal_scores, classes, cfg.target_classes)


def _objectness_roi_score(det: DetectionCache, roi: np.ndarray) -> float:
    obj = np.asarray(det.objectness_map, dtype=np.float32)
    if obj.size == 0:
        return 0.0
    if obj.ndim == 2:
        heat = obj
    elif obj.ndim >= 3:
        heat = obj.reshape(-1, obj.shape[-2], obj.shape[-1]).max(axis=0)
    else:
        side = int(round(float(np.sqrt(obj.size))))
        if side * side != obj.size:
            return 0.0
        heat = obj.reshape(side, side)
    if heat.ndim != 2 or heat.shape[0] == 0 or heat.shape[1] == 0:
        return 0.0
    grid_y, grid_x = heat.shape
    h, w = det.image_shape
    x1, y1, x2, y2 = np.asarray(roi, dtype=np.float32).reshape(4)
    gx1 = int(np.floor(np.clip(x1 / max(w, 1), 0.0, 1.0) * grid_x))
    gy1 = int(np.floor(np.clip(y1 / max(h, 1), 0.0, 1.0) * grid_y))
    gx2 = int(np.ceil(np.clip(x2 / max(w, 1), 0.0, 1.0) * grid_x))
    gy2 = int(np.ceil(np.clip(y2 / max(h, 1), 0.0, 1.0) * grid_y))
    gx1 = int(np.clip(gx1, 0, grid_x - 1))
    gy1 = int(np.clip(gy1, 0, grid_y - 1))
    gx2 = int(np.clip(max(gx2, gx1 + 1), 1, grid_x))
    gy2 = int(np.clip(max(gy2, gy1 + 1), 1, grid_y))
    window = heat[gy1:gy2, gx1:gx2]
    return float(np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).mean()) if window.size else 0.0


def _score_rois(
    rois: list[np.ndarray],
    det: DetectionCache,
    cfg: InferenceConfig,
    bench_cfg: BenchmarkConfig,
) -> np.ndarray:
    if not rois:
        return np.zeros((0,), dtype=np.float32)
    boxes, scores, _classes = _proposal_detections(det, cfg, bench_cfg)
    roi_arr = np.stack(rois).astype(np.float32)
    roi_scores = np.zeros((len(rois),), dtype=np.float32)
    if len(boxes) > 0:
        pts = centers(boxes)
        image_area = max(float(det.image_shape[0] * det.image_shape[1]), 1.0)
        box_area_ratio = area(boxes) / image_area
        small_bonus = (box_area_ratio <= 0.01).astype(np.float32) * 0.75
        quality = 0.25 + _proposal_quality(scores, bench_cfg) + small_bonus
        for index, roi in enumerate(roi_arr):
            x1, y1, x2, y2 = roi
            inside = (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
            if inside.any():
                roi_area = max(float(area(roi.reshape(1, 4))[0]), 1.0)
                density = float(roi_area / image_area)
                roi_scores[index] += float(quality[inside].sum()) / max(np.sqrt(density), 1e-3)
    for index, roi in enumerate(roi_arr):
        roi_scores[index] += 0.5 * _objectness_roi_score(det, roi)
    return roi_scores


def _select_top_rois(
    rois: list[np.ndarray],
    scores: np.ndarray,
    budget: int,
    overlap_threshold: float,
) -> list[int]:
    if budget <= 0 or not rois:
        return []
    roi_arr = np.stack(rois).astype(np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    order = np.argsort(scores)[::-1]
    selected: list[int] = []
    for idx in order:
        if len(selected) >= budget:
            break
        if selected:
            inter = intersection_matrix(roi_arr[[idx]], roi_arr[selected])[0]
            current_area = max(float(area(roi_arr[[idx]])[0]), 1.0)
            if float(np.max(inter / current_area)) >= float(overlap_threshold):
                continue
        selected.append(int(idx))
    if len(selected) < min(budget, len(rois)):
        for idx in order:
            if len(selected) >= budget:
                break
            if int(idx) not in selected:
                selected.append(int(idx))
    return selected


def _predict_from_crop_predictions(
    det: DetectionCache,
    cfg: InferenceConfig,
    full_predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
    crop_predictions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    selected_indices: list[int],
    gated: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    full_boxes, full_scores, full_classes = full_predictions
    boxes_parts = [full_boxes]
    scores_parts = [full_scores]
    classes_parts = [full_classes]
    accepted_count = 0
    for idx in selected_indices:
        boxes_i, scores_i, classes_i = crop_predictions[idx]
        classes_i = cfg.class_mapping.map_model_classes(classes_i)
        boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
        if gated:
            gain, utility, max_score = _new_detection_stats(
                full_boxes,
                full_scores,
                full_classes,
                boxes_parts[1:],
                scores_parts[1:],
                classes_parts[1:],
                boxes_i,
                scores_i,
                classes_i,
                det.image_shape,
                cfg.merge_iou,
                cfg.duplicate_iou,
            )
            if _crop_rejection_reason(len(boxes_i), gain, utility, max_score, cfg) is not None:
                continue
        boxes_parts.append(boxes_i)
        scores_parts.append(scores_i)
        classes_parts.append(classes_i)
        accepted_count += 1
    boxes, scores, classes = _merge_predictions(det.image_shape, cfg.merge_iou, boxes_parts, scores_parts, classes_parts)
    return boxes, scores, classes, accepted_count, len(selected_indices)


def _proposal_rois(
    det: DetectionCache,
    cfg: InferenceConfig,
    bench_cfg: BenchmarkConfig,
    budget: int,
) -> list[np.ndarray]:
    boxes, scores, _classes = _proposal_detections(det, cfg, bench_cfg)
    if len(boxes) == 0 or budget <= 0:
        return []
    h, w = det.image_shape
    side = max(1.0, min(h, w) * float(bench_cfg.fixed_slice_fraction))
    pts = centers(boxes)
    priorities = _proposal_quality(scores, bench_cfg)
    image_area = max(float(h * w), 1.0)
    priorities += (area(boxes) / image_area <= 0.01).astype(np.float32) * 0.75
    priorities += np.asarray([_objectness_roi_score(det, box) for box in boxes], dtype=np.float32) * 0.25
    order = np.argsort(priorities)[::-1]
    rois = [
        box_from_center(float(pts[idx, 0]), float(pts[idx, 1]), side, det.image_shape)
        for idx in order
        if priorities[idx] > 0.0
    ]
    scores_for_rois = np.asarray([priorities[idx] for idx in order if priorities[idx] > 0.0], dtype=np.float32)
    selected = _select_top_rois(rois, scores_for_rois, budget, bench_cfg.proposal_overlap_threshold)
    return [rois[idx] for idx in selected]


def _predict_rl_sahi(
    model: YOLO,
    policy,
    device_t: torch.device,
    image_path: Path,
    det: DetectionCache,
    cfg: InferenceConfig,
    env_cfg,
    state_cfg: StateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    full_boxes, full_scores, full_classes = _full_predictions(det, cfg)
    slice_boxes_all: list[np.ndarray] = []
    slice_scores_all: list[np.ndarray] = []
    slice_classes_all: list[np.ndarray] = []
    accepted_rois: list[np.ndarray] = []
    attempted_rois: list[np.ndarray] = []
    crop_inference_count = 0
    max_attempts = int(cfg.max_slice_attempts) if cfg.max_slice_attempts > 0 else int(env_cfg.max_slices * 2)
    crop_batch_size = max(int(cfg.crop_batch_size), 1)
    attempt_idx = 1
    stop_attempts = False
    consecutive_rejections = 0
    rejection_limit = max(int(cfg.max_consecutive_rejections), 0)
    while attempt_idx <= max_attempts and len(accepted_rois) < env_cfg.max_slices and not stop_attempts:
        remaining_attempts = max_attempts - attempt_idx + 1
        remaining_slices = max(int(env_cfg.max_slices) - len(accepted_rois), 1)
        pending_limit = min(crop_batch_size, remaining_attempts, remaining_slices)
        pending: list[tuple[np.ndarray, dict]] = []
        while (
            len(pending) < pending_limit
            and attempt_idx <= max_attempts
            and len(accepted_rois) < env_cfg.max_slices
        ):
            history_arr = (
                np.stack(attempted_rois).astype(np.float32)
                if attempted_rois
                else np.zeros((0, 4), dtype=np.float32)
            )
            overlap_arr = (
                np.stack(accepted_rois).astype(np.float32)
                if accepted_rois
                else np.zeros((0, 4), dtype=np.float32)
            )
            env = SliceEnv(
                det,
                None,
                env_cfg=env_cfg,
                state_cfg=state_cfg,
                previous_rois=history_arr,
                overlap_rois=overlap_arr,
                target_classes=cfg.target_classes,
                class_mapping=cfg.class_mapping,
            )
            roi, _actions, info = rollout_one_slice(policy, env, device_t)
            repeat_attempt_overlap = _attempt_overlap(roi, attempted_rois)
            attempted_rois.append(roi)
            if _skip_crop_reason(info, cfg) is not None:
                consecutive_rejections += 1
                if repeat_attempt_overlap >= 0.95:
                    stop_attempts = True
                    break
                if rejection_limit > 0 and consecutive_rejections >= rejection_limit:
                    stop_attempts = True
                    break
            else:
                pending.append((roi, info))
            attempt_idx += 1

        if not pending:
            continue

        predictions = run_yolo_on_crops(
            model,
            [image_path] * len(pending),
            [roi for roi, _info in pending],
            imgsz=cfg.slice_imgsz,
            conf=cfg.output_conf,
            iou=cfg.iou,
            max_det=cfg.max_det,
            device=cfg.device,
        )
        crop_inference_count += len(pending)
        for (roi, _info), (boxes_i, scores_i, classes_i) in zip(pending, predictions):
            classes_i = cfg.class_mapping.map_model_classes(classes_i)
            boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
            new_detection_gain, new_detection_utility, new_detection_max_score = _new_detection_stats(
                full_boxes,
                full_scores,
                full_classes,
                slice_boxes_all,
                slice_scores_all,
                slice_classes_all,
                boxes_i,
                scores_i,
                classes_i,
                det.image_shape,
                cfg.merge_iou,
                cfg.duplicate_iou,
            )
            if _crop_rejection_reason(
                len(boxes_i),
                new_detection_gain,
                new_detection_utility,
                new_detection_max_score,
                cfg,
            ) is not None:
                consecutive_rejections += 1
                continue
            consecutive_rejections = 0
            accepted_rois.append(roi)
            slice_boxes_all.append(boxes_i)
            slice_scores_all.append(scores_i)
            slice_classes_all.append(classes_i)
        if rejection_limit > 0 and consecutive_rejections >= rejection_limit:
            stop_attempts = True

    boxes, scores, classes = _merge_predictions(
        det.image_shape,
        cfg.merge_iou,
        [full_boxes, *slice_boxes_all],
        [full_scores, *slice_scores_all],
        [full_classes, *slice_classes_all],
    )
    return boxes, scores, classes, len(accepted_rois), crop_inference_count


def _ap_from_pr(tp: np.ndarray, fp: np.ndarray, total_gt: int) -> float:
    if total_gt == 0 or len(tp) == 0:
        return 0.0
    recall = np.cumsum(tp) / max(float(total_gt), 1.0)
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-9)
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    changed = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[changed + 1] - recall[changed]) * precision[changed + 1]))


def _ap_and_fp_at_iou(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ground_truth: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]],
    target_classes: tuple[int, ...],
    iou_threshold: float,
) -> tuple[float, int]:
    aps: list[float] = []
    total_fp = 0
    for cls in target_classes:
        pred_rows: list[tuple[str, float, np.ndarray]] = []
        gt_by_image: dict[str, np.ndarray] = {}
        matched_by_image: dict[str, np.ndarray] = {}
        for image_id, (gt_boxes, gt_classes, _shape) in ground_truth.items():
            gt_cls_boxes = gt_boxes[gt_classes.astype(np.int64) == int(cls)]
            gt_by_image[image_id] = gt_cls_boxes
            matched_by_image[image_id] = np.zeros((len(gt_cls_boxes),), dtype=bool)
        for image_id, (boxes, scores, classes) in predictions.items():
            mask = classes.astype(np.int64) == int(cls)
            for box, score in zip(boxes[mask], scores[mask]):
                pred_rows.append((image_id, float(score), box))
        pred_rows.sort(key=lambda row: row[1], reverse=True)
        tp = np.zeros((len(pred_rows),), dtype=np.float32)
        fp = np.zeros((len(pred_rows),), dtype=np.float32)
        for index, (image_id, _score, box) in enumerate(pred_rows):
            gt_boxes = gt_by_image[image_id]
            if len(gt_boxes) == 0:
                fp[index] = 1.0
                continue
            ious = iou_matrix(box.reshape(1, 4), gt_boxes)[0]

            # Mask out already matched ground truth boxes
            ious[matched_by_image[image_id]] = -1.0

            best = int(ious.argmax())
            if float(ious[best]) >= iou_threshold:
                tp[index] = 1.0
                matched_by_image[image_id][best] = True
            else:
                fp[index] = 1.0
        total_gt = sum(len(x) for x in gt_by_image.values())
        if total_gt > 0:
            aps.append(_ap_from_pr(tp, fp, total_gt))
        total_fp += int(fp.sum())
    return float(np.mean(aps)) if aps else 0.0, total_fp


def _precision_recall_at_iou(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ground_truth: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]],
    target_classes: tuple[int, ...],
    iou_threshold: float,
) -> tuple[float, float]:
    total_tp = 0
    total_fp = 0
    total_gt = 0
    for cls in target_classes:
        gt_by_image: dict[str, np.ndarray] = {}
        matched_by_image: dict[str, np.ndarray] = {}
        for image_id, (gt_boxes, gt_classes, _shape) in ground_truth.items():
            gt_cls_boxes = gt_boxes[gt_classes.astype(np.int64) == int(cls)]
            gt_by_image[image_id] = gt_cls_boxes
            matched_by_image[image_id] = np.zeros((len(gt_cls_boxes),), dtype=bool)
            total_gt += len(gt_cls_boxes)

        pred_rows: list[tuple[str, float, np.ndarray]] = []
        for image_id, (boxes, scores, classes) in predictions.items():
            mask = classes.astype(np.int64) == int(cls)
            pred_rows.extend(
                (image_id, float(score), box) for box, score in zip(boxes[mask], scores[mask])
            )
        pred_rows.sort(key=lambda row: row[1], reverse=True)

        for image_id, _score, box in pred_rows:
            gt_boxes = gt_by_image[image_id]
            if len(gt_boxes) == 0:
                total_fp += 1
                continue
            ious = iou_matrix(box.reshape(1, 4), gt_boxes)[0]
            ious[matched_by_image[image_id]] = -1.0
            best = int(ious.argmax())
            if float(ious[best]) >= iou_threshold:
                total_tp += 1
                matched_by_image[image_id][best] = True
            else:
                total_fp += 1

    precision = float(total_tp / max(total_tp + total_fp, 1))
    recall = float(total_tp / max(total_gt, 1))
    return precision, recall


def _small_recall_at_iou(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ground_truth: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]],
    iou_threshold: float,
    small_area_threshold: float,
) -> float:
    small_total = 0
    small_hit = 0
    for image_id, (gt_boxes, gt_classes, shape) in ground_truth.items():
        if len(gt_boxes) == 0:
            continue
        h, w = shape
        small_mask = (area(gt_boxes) / max(float(h * w), 1.0)) <= small_area_threshold
        small_total += int(small_mask.sum())
        if not small_mask.any():
            continue
        boxes, _scores, classes = predictions[image_id]
        for gt_box, gt_cls in zip(gt_boxes[small_mask], gt_classes[small_mask]):
            pred_mask = classes.astype(np.int64) == int(gt_cls)
            if pred_mask.any() and float(iou_matrix(gt_box.reshape(1, 4), boxes[pred_mask]).max()) >= iou_threshold:
                small_hit += 1
    return float(small_hit / max(small_total, 1))


def _evaluate_method(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ground_truth: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]],
    target_classes: tuple[int, ...],
    iou_threshold: float,
    small_area_threshold: float,
) -> dict[str, float]:
    ap_thresholds = tuple(float(x) for x in np.arange(0.50, 0.96, 0.05))
    per_class: dict[int, dict[float, tuple[float, int]]] = {}
    for cls in target_classes:
        per_class[int(cls)] = {
            threshold: _ap_and_fp_at_iou(
                predictions, ground_truth, (int(cls),), threshold
            )
            for threshold in ap_thresholds
        }

    active_classes = [
        int(cls)
        for cls in target_classes
        if any(
            bool(np.any(gt_classes.astype(np.int64) == int(cls)))
            for _gt_boxes, gt_classes, _shape in ground_truth.values()
        )
    ]

    def mean_ap(threshold: float) -> float:
        values = [per_class[cls][threshold][0] for cls in active_classes]
        return float(np.mean(values)) if values else 0.0

    threshold_50 = ap_thresholds[0]
    threshold_75 = ap_thresholds[5]
    ap_values = [mean_ap(threshold) for threshold in ap_thresholds]
    ap50 = mean_ap(threshold_50)
    ap75 = mean_ap(threshold_75)
    primary_threshold = min(ap_thresholds, key=lambda value: abs(value - float(iou_threshold)))
    if not np.isclose(primary_threshold, float(iou_threshold)):
        primary_rows = {
            int(cls): _ap_and_fp_at_iou(
                predictions, ground_truth, (int(cls),), float(iou_threshold)
            )
            for cls in target_classes
        }
        total_fp = sum(row[1] for row in primary_rows.values())
    else:
        total_fp = sum(
            per_class[int(cls)][primary_threshold][1] for cls in target_classes
        )
    precision, recall = _precision_recall_at_iou(
        predictions, ground_truth, target_classes, iou_threshold
    )
    metrics = {
        "AP": float(np.mean(ap_values)) if ap_values else 0.0,
        "AP50": ap50,
        "AP75": ap75,
        "mAP50": ap50,
        "precision": precision,
        "recall": recall,
        "small_recall": _small_recall_at_iou(predictions, ground_truth, iou_threshold, small_area_threshold),
        "fp_per_image": float(total_fp / max(len(ground_truth), 1)),
    }
    for cls in target_classes:
        class_ap_values = [per_class[int(cls)][threshold][0] for threshold in ap_thresholds]
        class_ap50, class_fp50 = per_class[int(cls)][threshold_50]
        metrics[f"AP_class_{int(cls)}"] = float(np.mean(class_ap_values))
        metrics[f"AP50_class_{int(cls)}"] = float(class_ap50)
        metrics[f"FP50_class_{int(cls)}_per_image"] = float(
            class_fp50 / max(len(ground_truth), 1)
        )
    return metrics


def evaluate_rl_sahi_policy(
    model: YOLO,
    policy,
    device_t: torch.device,
    weights: Path,
    images: list[Path],
    image_root: Path,
    label_root: Path,
    cache_root: Path,
    split: str,
    infer_cfg: InferenceConfig,
    bench_cfg: BenchmarkConfig,
    env_cfg,
    state_cfg: StateConfig,
    use_cache: bool = True,
) -> dict[str, float]:
    if not images:
        raise FileNotFoundError(f"No images provided for split '{split}'")

    infer_cfg = replace(
        infer_cfg,
        target_classes=bench_cfg.target_classes,
        class_mapping=bench_cfg.class_mapping,
    )
    small_threshold = _resolve_small_area_threshold(images, image_root, label_root, bench_cfg)
    ground_truth: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    crops: list[int] = []
    accepted_crops: list[int] = []
    latency: list[float] = []

    for image_index, image_path in enumerate(images, start=1):
        if image_index == 1 or image_index % 25 == 0 or image_index == len(images):
            print(
                f"[benchmark] {split}: image {image_index}/{len(images)} ({image_path.name})",
                flush=True,
            )
        image_id = image_path.stem
        gt_boxes, gt_classes = _read_gt(
            image_path,
            image_root,
            label_root,
            bench_cfg.target_classes,
            bench_cfg.class_mapping,
        )
        shape = _image_shape(image_path)
        ground_truth[image_id] = (gt_boxes, gt_classes, shape)

        det = get_initial_detection(
            model=model,
            weights=weights,
            image_path=image_path,
            weights_imgsz=infer_cfg.full_imgsz,
            full_conf=infer_cfg.full_conf,
            full_iou=infer_cfg.iou,
            max_det=infer_cfg.max_det,
            device=infer_cfg.device,
            feature_layers=infer_cfg.feature_layers,
            aux_grid_size=state_cfg.grid_size,
            spatial_feature_channels=state_cfg.spatial_feature_channels,
            cache_root=cache_root,
            split=split,
            use_cache=use_cache,
        )

        start = time.perf_counter()
        boxes, scores, classes, accepted_crop_count, crop_count = _predict_rl_sahi(
            model,
            policy,
            device_t,
            image_path,
            det,
            infer_cfg,
            env_cfg,
            state_cfg,
        )
        predictions[image_id] = (boxes, scores, classes)
        latency.append(time.perf_counter() - start)
        crops.append(crop_count)
        accepted_crops.append(accepted_crop_count)

    metrics = _evaluate_method(
        predictions,
        ground_truth,
        bench_cfg.target_classes,
        bench_cfg.iou_threshold,
        small_threshold,
    )
    return {
        **metrics,
        "crops_per_image": float(np.mean(crops)),
        "accepted_crops_per_image": float(np.mean(accepted_crops)),
        "latency_ms_per_image": float(np.mean(latency) * 1000.0),
        "images": float(len(images)),
        "small_area_threshold": small_threshold,
    }


def benchmark_split(
    weights: Path,
    checkpoint: Path,
    image_root: Path,
    label_root: Path,
    cache_root: Path,
    split: str,
    infer_cfg: InferenceConfig,
    bench_cfg: BenchmarkConfig,
    out_dir: Path,
    limit: int | None = None,
    use_cache: bool = True,
) -> list[dict[str, float | str]]:
    images = select_benchmark_images(
        iter_images(image_root, split=split),
        limit,
        sampling=bench_cfg.sampling,
        seed=bench_cfg.seed,
    )
    if not images:
        raise FileNotFoundError(f"No images found for split '{split}'")

    infer_cfg = replace(
        infer_cfg,
        target_classes=bench_cfg.target_classes,
        class_mapping=bench_cfg.class_mapping,
    )
    small_threshold = _resolve_small_area_threshold(images, image_root, label_root, bench_cfg)
    model = load_yolo(weights, device=infer_cfg.device)
    device_t = resolve_torch_device(infer_cfg.device)
    policy, checkpoint_data = load_policy(checkpoint, device_t)
    env_cfg = checkpoint_data["env_cfg_obj"]
    state_cfg = checkpoint_data.get("state_cfg_obj", StateConfig())

    budgeted_counts = tuple(sorted({int(x) for x in bench_cfg.budgeted_crop_counts if int(x) > 0}))
    budgeted_methods = {budget: f"fixed_grid_budget_{budget}" for budget in budgeted_counts}
    gated_budgeted_methods = (
        {budget: f"fixed_grid_budget_{budget}_gated" for budget in budgeted_counts}
        if bench_cfg.include_gated_variants
        else {}
    )
    proposal_method = f"proposal_sahi_{int(bench_cfg.proposal_crop_count)}"
    proposal_gated_method = f"{proposal_method}_gated"
    method_names = ["yolo_full"]
    if bench_cfg.include_fixed_grid_full:
        method_names.append("fixed_grid_sahi")
        if bench_cfg.include_gated_variants:
            method_names.append("fixed_grid_sahi_gated")
    method_names.extend(budgeted_methods.values())
    method_names.extend(gated_budgeted_methods.values())
    if int(bench_cfg.proposal_crop_count) > 0:
        method_names.append(proposal_method)
        if bench_cfg.include_gated_variants:
            method_names.append(proposal_gated_method)
    method_names.append("rl_sahi")

    ground_truth: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    predictions = {name: {} for name in method_names}
    crops = {key: [] for key in predictions}
    accepted_crops = {key: [] for key in predictions}
    latency = {key: [] for key in predictions}
    initial_state_latency: list[float] = []
    warmup_images = _effective_warmup_images(len(images), bench_cfg.warmup_images)
    timed_images = len(images) - warmup_images
    if warmup_images:
        print(
            f"[benchmark] {split}: warming up on the first {warmup_images} images; "
            f"latency uses the remaining {timed_images}",
            flush=True,
        )

    for image_index, image_path in enumerate(images, start=1):
        measure_latency = image_index > warmup_images
        if image_index == 1 or image_index % 25 == 0 or image_index == len(images):
            print(
                f"[benchmark] {split}: image {image_index}/{len(images)} ({image_path.name})",
                flush=True,
            )
        image_id = image_path.stem
        gt_boxes, gt_classes = _read_gt(
            image_path,
            image_root,
            label_root,
            bench_cfg.target_classes,
            bench_cfg.class_mapping,
        )
        shape = _image_shape(image_path)
        ground_truth[image_id] = (gt_boxes, gt_classes, shape)

        initial_start = time.perf_counter()
        det = get_initial_detection(
            model=model,
            weights=weights,
            image_path=image_path,
            weights_imgsz=infer_cfg.full_imgsz,
            full_conf=infer_cfg.full_conf,
            full_iou=infer_cfg.iou,
            max_det=infer_cfg.max_det,
            device=infer_cfg.device,
            feature_layers=infer_cfg.feature_layers,
            aux_grid_size=state_cfg.grid_size,
            spatial_feature_channels=state_cfg.spatial_feature_channels,
            cache_root=cache_root,
            split=split,
            use_cache=use_cache,
        )
        if measure_latency:
            initial_state_latency.append(time.perf_counter() - initial_start)

        full_predictions = _full_predictions(det, infer_cfg)

        start = time.perf_counter()
        predictions["yolo_full"][image_id] = full_predictions
        if measure_latency:
            latency["yolo_full"].append(time.perf_counter() - start)
        crops["yolo_full"].append(0)
        accepted_crops["yolo_full"].append(0)

        fixed_rois = _fixed_grid_rois(det.image_shape, bench_cfg.fixed_slice_fraction, bench_cfg.fixed_overlap)
        fixed_scores = _score_rois(fixed_rois, det, infer_cfg, bench_cfg)

        if bench_cfg.include_fixed_grid_full:
            start = time.perf_counter()
            fixed_crop_predictions = run_yolo_on_crops(
                model,
                [image_path] * len(fixed_rois),
                fixed_rois,
                imgsz=infer_cfg.slice_imgsz,
                conf=infer_cfg.output_conf,
                iou=infer_cfg.iou,
                max_det=infer_cfg.max_det,
                device=infer_cfg.device,
            )
            fixed_crop_latency = time.perf_counter() - start
            post_start = time.perf_counter()
            boxes, scores, classes, accepted_crop_count, crop_count = _predict_from_crop_predictions(
                det,
                infer_cfg,
                full_predictions,
                fixed_crop_predictions,
                list(range(len(fixed_rois))),
            )
            predictions["fixed_grid_sahi"][image_id] = (boxes, scores, classes)
            if measure_latency:
                latency["fixed_grid_sahi"].append(
                    fixed_crop_latency + time.perf_counter() - post_start
                )
            crops["fixed_grid_sahi"].append(crop_count)
            accepted_crops["fixed_grid_sahi"].append(accepted_crop_count)
            if bench_cfg.include_gated_variants:
                post_start = time.perf_counter()
                boxes, scores, classes, accepted_crop_count, crop_count = _predict_from_crop_predictions(
                    det,
                    infer_cfg,
                    full_predictions,
                    fixed_crop_predictions,
                    list(range(len(fixed_rois))),
                    gated=True,
                )
                predictions["fixed_grid_sahi_gated"][image_id] = (boxes, scores, classes)
                if measure_latency:
                    latency["fixed_grid_sahi_gated"].append(
                        fixed_crop_latency + time.perf_counter() - post_start
                    )
                crops["fixed_grid_sahi_gated"].append(crop_count)
                accepted_crops["fixed_grid_sahi_gated"].append(accepted_crop_count)

        for budget, method in budgeted_methods.items():
            selected = _select_top_rois(
                fixed_rois,
                fixed_scores,
                budget,
                bench_cfg.proposal_overlap_threshold,
            )
            selected_rois = [fixed_rois[idx] for idx in selected]
            start = time.perf_counter()
            crop_predictions = run_yolo_on_crops(
                model,
                [image_path] * len(selected_rois),
                selected_rois,
                imgsz=infer_cfg.slice_imgsz,
                conf=infer_cfg.output_conf,
                iou=infer_cfg.iou,
                max_det=infer_cfg.max_det,
                device=infer_cfg.device,
            )
            crop_latency = time.perf_counter() - start
            post_start = time.perf_counter()
            boxes, scores, classes, accepted_crop_count, crop_count = _predict_from_crop_predictions(
                det,
                infer_cfg,
                full_predictions,
                crop_predictions,
                list(range(len(selected_rois))),
            )
            predictions[method][image_id] = (boxes, scores, classes)
            if measure_latency:
                latency[method].append(crop_latency + time.perf_counter() - post_start)
            crops[method].append(crop_count)
            accepted_crops[method].append(accepted_crop_count)
            if bench_cfg.include_gated_variants:
                gated_method = gated_budgeted_methods[budget]
                post_start = time.perf_counter()
                boxes, scores, classes, accepted_crop_count, crop_count = _predict_from_crop_predictions(
                    det,
                    infer_cfg,
                    full_predictions,
                    crop_predictions,
                    list(range(len(selected_rois))),
                    gated=True,
                )
                predictions[gated_method][image_id] = (boxes, scores, classes)
                if measure_latency:
                    latency[gated_method].append(crop_latency + time.perf_counter() - post_start)
                crops[gated_method].append(crop_count)
                accepted_crops[gated_method].append(accepted_crop_count)

        if int(bench_cfg.proposal_crop_count) > 0:
            proposal_rois = _proposal_rois(det, infer_cfg, bench_cfg, int(bench_cfg.proposal_crop_count))
            start = time.perf_counter()
            proposal_crop_predictions = run_yolo_on_crops(
                model,
                [image_path] * len(proposal_rois),
                proposal_rois,
                imgsz=infer_cfg.slice_imgsz,
                conf=infer_cfg.output_conf,
                iou=infer_cfg.iou,
                max_det=infer_cfg.max_det,
                device=infer_cfg.device,
            )
            proposal_crop_latency = time.perf_counter() - start
            post_start = time.perf_counter()
            boxes, scores, classes, accepted_crop_count, crop_count = _predict_from_crop_predictions(
                det,
                infer_cfg,
                full_predictions,
                proposal_crop_predictions,
                list(range(len(proposal_rois))),
            )
            predictions[proposal_method][image_id] = (boxes, scores, classes)
            if measure_latency:
                latency[proposal_method].append(
                    proposal_crop_latency + time.perf_counter() - post_start
                )
            crops[proposal_method].append(crop_count)
            accepted_crops[proposal_method].append(accepted_crop_count)
            if bench_cfg.include_gated_variants:
                post_start = time.perf_counter()
                boxes, scores, classes, accepted_crop_count, crop_count = _predict_from_crop_predictions(
                    det,
                    infer_cfg,
                    full_predictions,
                    proposal_crop_predictions,
                    list(range(len(proposal_rois))),
                    gated=True,
                )
                predictions[proposal_gated_method][image_id] = (boxes, scores, classes)
                if measure_latency:
                    latency[proposal_gated_method].append(
                        proposal_crop_latency + time.perf_counter() - post_start
                    )
                crops[proposal_gated_method].append(crop_count)
                accepted_crops[proposal_gated_method].append(accepted_crop_count)

        start = time.perf_counter()
        boxes, scores, classes, accepted_crop_count, crop_count = _predict_rl_sahi(
            model, policy, device_t, image_path, det, infer_cfg, env_cfg, state_cfg
        )
        predictions["rl_sahi"][image_id] = (boxes, scores, classes)
        if measure_latency:
            latency["rl_sahi"].append(time.perf_counter() - start)
        crops["rl_sahi"].append(crop_count)
        accepted_crops["rl_sahi"].append(accepted_crop_count)

    rows: list[dict[str, float | str]] = []
    mean_initial_ms = float(np.mean(initial_state_latency) * 1000.0)
    for method, method_predictions in predictions.items():
        print(f"[benchmark] {split}: evaluating {method}", flush=True)
        metrics = _evaluate_method(
            method_predictions,
            ground_truth,
            bench_cfg.target_classes,
            bench_cfg.iou_threshold,
            small_threshold,
        )
        crop_mean = float(np.mean(crops[method]))
        incremental_ms = float(np.mean(latency[method]) * 1000.0)
        end_to_end_ms = mean_initial_ms + incremental_ms
        rows.append(
            {
                "method": method,
                **metrics,
                "crops_per_image": crop_mean,
                "accepted_crops_per_image": float(np.mean(accepted_crops[method])),
                "acceptance_rate": float(
                    np.sum(accepted_crops[method]) / max(float(np.sum(crops[method])), 1.0)
                ),
                "latency_ms_per_image": incremental_ms,
                "initial_state_ms_per_image": mean_initial_ms,
                "end_to_end_ms_per_image": end_to_end_ms,
                "images_per_second": float(1000.0 / max(end_to_end_ms, 1e-9)),
                "detector_calls_per_image": 1.0 + crop_mean,
                "effective_gflops": float(
                    bench_cfg.agent_gflops + bench_cfg.detector_gflops * (1.0 + crop_mean)
                ),
                "images": float(len(images)),
                "warmup_images": float(warmup_images),
                "timed_images": float(timed_images),
                "small_area_threshold": small_threshold,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(weights).resolve().parent
    manifest = {
        "split": split,
        "use_cache": bool(use_cache),
        "device": str(device_t),
        "python": platform.python_version(),
        "git_revision": _git_revision(project_root),
        "weights": file_fingerprint(weights),
        "checkpoint": file_fingerprint(checkpoint),
        "images": [image.name for image in images],
        "warmup_images": warmup_images,
        "timed_images": timed_images,
        "inference_config": asdict(infer_cfg),
        "benchmark_config": asdict(bench_cfg),
    }
    (out_dir / "benchmark.json").write_text(
        json.dumps({"config": asdict(bench_cfg), "manifest": manifest, "results": rows}, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "benchmark.csv").open("w", encoding="utf-8") as f:
        header = list(rows[0].keys())
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row[key]) for key in header) + "\n")
    return rows
