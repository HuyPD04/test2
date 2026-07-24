from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.cache import file_fingerprint
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import iter_images, read_image
from rl_sahi.common.device import print_device_info, resolve_torch_device
from rl_sahi.common.boxes import iou_matrix
from rl_sahi.detection.yolo import load_yolo
from rl_sahi.eval.benchmark import (
    BenchmarkConfig,
    _evaluate_method,
    _full_predictions,
    _read_gt,
    _resolve_small_area_threshold,
    select_benchmark_images,
)
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crops
from rl_sahi.inference.merge import merge_predictions_with_sources
from rl_sahi.inference.pipeline import (
    _attempt_overlap,
    _crop_rejection_reason,
    _filter_classes,
    _new_detection_stats,
    _skip_crop_reason,
    filter_boundary_boxes,
    get_initial_detection,
)
from rl_sahi.inference.roi_prefilter import score_roi_candidates, select_roi_candidates
from rl_sahi.inference.rollout import rollout_one_slice
from rl_sahi.rl.checkpoint import load_policy
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig


AP_THRESHOLDS = tuple(float(x) for x in np.arange(0.50, 0.96, 0.05))


def _bool_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _int_tuple(value) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def _optional_float(value, default: float | None = None) -> float | None:
    raw = default if value is None else value
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null", "false", "off"}:
        return None
    return float(raw)


def _path_or_none(cfg, key: str) -> Path | None:
    try:
        return cfg.path_value(key)
    except KeyError:
        return None


def _build_configs(cfg, policy_device: str | None) -> tuple[InferenceConfig, BenchmarkConfig]:
    infer_cfg = cfg.section("infer")
    benchmark_cfg = cfg.section("benchmark")
    target_classes = cfg.target_classes()
    class_mapping = ClassMapping.from_config(cfg.section("classes"))
    device = cfg.optional_str("infer", "device")
    output_conf = float(benchmark_cfg.get("output_conf", infer_cfg["output_conf"]))
    inference = InferenceConfig(
        full_imgsz=int(infer_cfg["full_imgsz"]),
        slice_imgsz=int(infer_cfg["slice_imgsz"]),
        full_conf=float(infer_cfg["full_conf"]),
        output_conf=output_conf,
        iou=float(infer_cfg["iou"]),
        merge_iou=float(infer_cfg["merge_iou"]),
        max_det=int(infer_cfg["max_det"]),
        device=device,
        policy_device=policy_device,
        feature_layers=cfg.feature_layers("infer"),
        min_slice_detections=int(infer_cfg.get("min_slice_detections", 1)),
        min_slice_utility=float(infer_cfg.get("min_slice_utility", 0.5)),
        min_new_detection_score=float(infer_cfg.get("min_new_detection_score", 0.45)),
        duplicate_iou=float(infer_cfg.get("duplicate_iou", infer_cfg.get("merge_iou", 0.5))),
        cross_class_duplicate_iou=_optional_float(infer_cfg.get("cross_class_duplicate_iou"), 0.85),
        cross_class_duplicate_ios=_optional_float(infer_cfg.get("cross_class_duplicate_ios"), 0.95),
        max_slice_attempts=int(infer_cfg.get("max_slice_attempts", 0)),
        roi_prefilter_enabled=_bool_value(infer_cfg.get("roi_prefilter_enabled", False)),
        roi_prefilter_topk=int(infer_cfg.get("roi_prefilter_topk", 3)),
        crop_batch_size=int(infer_cfg.get("crop_batch_size", 1)),
        max_consecutive_rejections=int(infer_cfg.get("max_consecutive_rejections", 0)),
        target_classes=target_classes,
        require_stop_for_acceptance=_bool_value(infer_cfg.get("require_stop_for_acceptance", True)),
        save_predictions=False,
        save_metadata=False,
        save_visualization=False,
        batched_inference=_bool_value(infer_cfg.get("batched_inference", False)),
        use_wbf=_bool_value(infer_cfg.get("use_wbf", False)),
        nms_type=str(infer_cfg.get("nms_type", "standard")),
        class_mapping=class_mapping,
    )
    benchmark = BenchmarkConfig(
        output_conf=output_conf,
        iou_threshold=float(benchmark_cfg.get("iou_threshold", 0.5)),
        fixed_slice_fraction=float(benchmark_cfg.get("fixed_slice_fraction", 0.35)),
        fixed_overlap=float(benchmark_cfg.get("fixed_overlap", 0.2)),
        budgeted_crop_counts=_int_tuple(benchmark_cfg.get("budgeted_crop_counts", (4, 8, 12))),
        include_fixed_grid_full=_bool_value(benchmark_cfg.get("include_fixed_grid_full", True)),
        include_sahi_library=_bool_value(benchmark_cfg.get("include_sahi_library", False)),
        sahi_model_type=str(benchmark_cfg.get("sahi_model_type", "yolov8")),
        sahi_slice_height=int(benchmark_cfg.get("sahi_slice_height", 640)),
        sahi_slice_width=int(benchmark_cfg.get("sahi_slice_width", 640)),
        sahi_overlap=float(benchmark_cfg.get("sahi_overlap", benchmark_cfg.get("fixed_overlap", 0.2))),
        sahi_perform_standard_pred=_bool_value(benchmark_cfg.get("sahi_perform_standard_pred", True)),
        sahi_auto_slice_resolution=_bool_value(benchmark_cfg.get("sahi_auto_slice_resolution", False)),
        sahi_postprocess_type=str(benchmark_cfg.get("sahi_postprocess_type", "GREEDYNMM")),
        sahi_postprocess_match_metric=str(benchmark_cfg.get("sahi_postprocess_match_metric", "IOS")),
        sahi_postprocess_match_threshold=float(benchmark_cfg.get("sahi_postprocess_match_threshold", 0.5)),
        include_gated_variants=_bool_value(benchmark_cfg.get("include_gated_variants", True)),
        proposal_crop_count=int(benchmark_cfg.get("proposal_crop_count", 8)),
        proposal_min_conf=float(benchmark_cfg.get("proposal_min_conf", 0.01)),
        proposal_max_conf=float(benchmark_cfg.get("proposal_max_conf", 0.5)),
        proposal_peak_conf=float(benchmark_cfg.get("proposal_peak_conf", 0.25)),
        proposal_overlap_threshold=float(benchmark_cfg.get("proposal_overlap_threshold", 0.7)),
        small_area_ratio=(
            None if benchmark_cfg.get("small_area_ratio") in (None, "")
            else float(benchmark_cfg["small_area_ratio"])
        ),
        small_area_percentile=(
            None if benchmark_cfg.get("small_area_percentile") in (None, "")
            else float(benchmark_cfg["small_area_percentile"])
        ),
        sampling=str(benchmark_cfg.get("sampling", "stratified")),
        seed=int(benchmark_cfg.get("seed", 42)),
        warmup_images=int(benchmark_cfg.get("warmup_images", 10)),
        detector_gflops=float(benchmark_cfg.get("detector_gflops", 21.5)),
        agent_gflops=float(benchmark_cfg.get("agent_gflops", 0.0)),
        eval_max_detections=int(benchmark_cfg.get("eval_max_detections", 500)),
        ignore_overlap_threshold=float(benchmark_cfg.get("ignore_overlap_threshold", 0.5)),
        target_classes=target_classes,
        class_mapping=class_mapping,
    )
    return inference, benchmark


def _concat_parts(parts: list[np.ndarray], shape: tuple[int, ...], dtype) -> np.ndarray:
    return np.concatenate(parts, axis=0) if parts else np.zeros(shape, dtype=dtype)


def _merge_from_parts(
    image_shape: tuple[int, int],
    cfg: InferenceConfig,
    boxes_parts: list[np.ndarray],
    scores_parts: list[np.ndarray],
    classes_parts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sources_parts = [
        np.full((len(boxes),), 0 if index == 0 else index, dtype=np.int32)
        for index, boxes in enumerate(boxes_parts)
    ]
    return merge_predictions_with_sources(
        image_shape,
        cfg.merge_iou,
        boxes_parts,
        scores_parts,
        classes_parts,
        sources_parts,
        cross_class_duplicate_iou=cfg.cross_class_duplicate_iou,
        cross_class_duplicate_ios=cfg.cross_class_duplicate_ios,
        use_wbf=cfg.use_wbf,
        nms_type=cfg.nms_type,
    )


def _cluster_indices(boxes: np.ndarray, classes: np.ndarray, iou_threshold: float) -> list[np.ndarray]:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    classes_i = np.asarray(classes, dtype=np.float32).reshape(-1).astype(np.int64)
    if len(boxes) == 0:
        return []

    clusters: list[np.ndarray] = []
    for cls in np.unique(classes_i):
        cls_idx = np.flatnonzero(classes_i == int(cls))
        if len(cls_idx) == 1:
            clusters.append(cls_idx)
            continue
        overlaps = iou_matrix(boxes[cls_idx], boxes[cls_idx]) >= float(iou_threshold)
        visited = np.zeros((len(cls_idx),), dtype=bool)
        for start in range(len(cls_idx)):
            if visited[start]:
                continue
            stack = [start]
            members: list[int] = []
            visited[start] = True
            while stack:
                current = stack.pop()
                members.append(current)
                neighbors = np.flatnonzero(overlaps[current] & ~visited)
                for neighbor in neighbors:
                    visited[int(neighbor)] = True
                    stack.append(int(neighbor))
            clusters.append(cls_idx[np.asarray(members, dtype=np.int64)])
    return clusters


def _oracle_cluster_predictions(
    image_shape: tuple[int, int],
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    cluster_iou: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del image_shape
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    out_boxes: list[np.ndarray] = []
    out_scores: list[float] = []
    out_classes: list[float] = []
    gt_classes_i = np.asarray(gt_classes, dtype=np.float32).reshape(-1).astype(np.int64)
    for cluster in _cluster_indices(boxes, classes, cluster_iou):
        cls = int(classes[cluster[0]])
        gt_mask = gt_classes_i == cls
        chosen = int(cluster[np.argmax(scores[cluster])])
        if gt_mask.any():
            cluster_ious = iou_matrix(boxes[cluster], gt_boxes[gt_mask])
            best_local = int(cluster_ious.max(axis=1).argmax())
            chosen = int(cluster[best_local])
        out_boxes.append(boxes[chosen])
        out_scores.append(float(scores[cluster].max()))
        out_classes.append(float(cls))

    order = np.argsort(np.asarray(out_scores, dtype=np.float32))[::-1]
    return (
        np.stack(out_boxes).astype(np.float32)[order],
        np.asarray(out_scores, dtype=np.float32)[order],
        np.asarray(out_classes, dtype=np.float32)[order],
    )


def _oracle_per_gt_predictions(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes_i = np.asarray(classes, dtype=np.float32).reshape(-1).astype(np.int64)
    if len(boxes) == 0 or len(gt_boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    chosen_boxes: list[np.ndarray] = []
    chosen_scores: list[float] = []
    chosen_classes: list[float] = []
    used: set[int] = set()
    candidates: list[tuple[float, int, int]] = []
    for gt_idx, gt_cls in enumerate(gt_classes.astype(np.int64)):
        pred_idx = np.flatnonzero(classes_i == int(gt_cls))
        if len(pred_idx) == 0:
            continue
        ious = iou_matrix(gt_boxes[gt_idx].reshape(1, 4), boxes[pred_idx])[0]
        best_local = int(ious.argmax())
        candidates.append((float(ious[best_local]), gt_idx, int(pred_idx[best_local])))
    for _iou, gt_idx, pred_idx in sorted(candidates, key=lambda item: item[0], reverse=True):
        if pred_idx in used:
            continue
        used.add(pred_idx)
        chosen_boxes.append(boxes[pred_idx])
        chosen_scores.append(max(1.0 - len(chosen_scores) * 1e-6, float(scores[pred_idx])))
        chosen_classes.append(float(gt_classes[gt_idx]))
    if not chosen_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.stack(chosen_boxes).astype(np.float32),
        np.asarray(chosen_scores, dtype=np.float32),
        np.asarray(chosen_classes, dtype=np.float32),
    )


def _candidate_recall(
    candidate_boxes: np.ndarray,
    candidate_classes: np.ndarray,
    ground_truth: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray]],
    image_id: str,
    totals: dict[float, list[int]],
) -> None:
    gt_boxes, gt_classes, _shape, _ignore_boxes = ground_truth[image_id]
    candidate_classes_i = np.asarray(candidate_classes, dtype=np.float32).reshape(-1).astype(np.int64)
    for threshold in AP_THRESHOLDS:
        hit = 0
        for gt_box, gt_cls in zip(gt_boxes, gt_classes.astype(np.int64)):
            pred_idx = np.flatnonzero(candidate_classes_i == int(gt_cls))
            if len(pred_idx) == 0:
                continue
            ious = iou_matrix(gt_box.reshape(1, 4), candidate_boxes[pred_idx])[0]
            if len(ious) > 0 and float(ious.max()) >= threshold:
                hit += 1
        totals[threshold][0] += int(hit)
        totals[threshold][1] += int(len(gt_boxes))


def _collect_parts(
    model,
    full_model,
    crop_model,
    policy,
    device_t: torch.device,
    image_path: Path,
    det,
    cfg: InferenceConfig,
    env_cfg,
    state_cfg: StateConfig,
    source_image: np.ndarray,
) -> dict[str, list[np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]]:
    env_cfg = type(env_cfg)(**{**asdict(env_cfg), "use_gpu_box_ops": False})
    env_static = SliceEnv.build_static_context(det, state_cfg, cfg.target_classes, cfg.class_mapping)
    full_boxes, full_scores, full_classes = _full_predictions(
        det, cfg, image_path=image_path, full_model=full_model, model=model
    )
    accepted_boxes: list[np.ndarray] = []
    accepted_scores: list[np.ndarray] = []
    accepted_classes: list[np.ndarray] = []
    candidate_boxes: list[np.ndarray] = []
    candidate_scores: list[np.ndarray] = []
    candidate_classes: list[np.ndarray] = []
    accepted_rois: list[np.ndarray] = []
    attempted_rois: list[np.ndarray] = []
    max_attempts = int(cfg.max_slice_attempts) if cfg.max_slice_attempts > 0 else int(env_cfg.max_slices * 2)

    def accept_prediction(roi: np.ndarray, prediction) -> None:
        boxes_i, scores_i, classes_i = prediction
        classes_i = cfg.class_mapping.map_model_classes(classes_i)
        boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
        boxes_i, scores_i, classes_i = filter_boundary_boxes(boxes_i, scores_i, classes_i, roi, det.image_shape)
        candidate_boxes.append(boxes_i)
        candidate_scores.append(scores_i)
        candidate_classes.append(classes_i)
        gain, utility, max_score = _new_detection_stats(
            full_boxes,
            full_scores,
            full_classes,
            accepted_boxes,
            accepted_scores,
            accepted_classes,
            boxes_i,
            scores_i,
            classes_i,
            det.image_shape,
            cfg.merge_iou,
            cfg.duplicate_iou,
            cfg.cross_class_duplicate_iou,
            cfg.cross_class_duplicate_ios,
            use_wbf=cfg.use_wbf,
            nms_type=cfg.nms_type,
        )
        if _crop_rejection_reason(len(boxes_i), gain, utility, max_score, cfg) is not None:
            return
        accepted_rois.append(roi)
        accepted_boxes.append(boxes_i)
        accepted_scores.append(scores_i)
        accepted_classes.append(classes_i)

    if cfg.batched_inference:
        rois: list[np.ndarray] = []
        attempt_idx = 1
        while attempt_idx <= max_attempts and len(rois) < env_cfg.max_slices:
            history_arr = (
                np.stack(attempted_rois).astype(np.float32)
                if attempted_rois
                else np.zeros((0, 4), dtype=np.float32)
            )
            env = SliceEnv(
                det,
                None,
                env_cfg=env_cfg,
                state_cfg=state_cfg,
                previous_rois=history_arr,
                overlap_rois=history_arr,
                target_classes=cfg.target_classes,
                class_mapping=cfg.class_mapping,
                static_context=env_static,
            )
            roi, _actions, info = rollout_one_slice(policy, env, device_t)
            repeat_attempt_overlap = _attempt_overlap(roi, attempted_rois)
            attempted_rois.append(roi)
            if _skip_crop_reason(info, cfg) is not None:
                if repeat_attempt_overlap >= 0.95:
                    break
            else:
                rois.append(roi)
            attempt_idx += 1
        if cfg.roi_prefilter_enabled and rois:
            roi_scores = score_roi_candidates(
                det,
                rois,
                state_cfg,
                cfg.target_classes,
                cfg.class_mapping,
            )
            selected = select_roi_candidates(roi_scores, cfg.roi_prefilter_topk)
            rois = [rois[index] for index in selected]
        if rois:
            predictions = run_yolo_on_crops(
                crop_model,
                [image_path] * len(rois),
                rois,
                imgsz=cfg.slice_imgsz,
                conf=cfg.output_conf,
                iou=cfg.iou,
                max_det=cfg.max_det,
                device=cfg.device,
                source_image=source_image,
            )
            for roi, prediction in zip(rois, predictions):
                accept_prediction(roi, prediction)
    else:
        crop_batch_size = max(int(cfg.crop_batch_size), 1)
        attempt_idx = 1
        stop_attempts = False
        consecutive_rejections = 0
        rejection_limit = max(int(cfg.max_consecutive_rejections), 0)
        while attempt_idx <= max_attempts and len(accepted_rois) < env_cfg.max_slices and not stop_attempts:
            pending: list[np.ndarray] = []
            pending_limit = min(crop_batch_size, max_attempts - attempt_idx + 1, env_cfg.max_slices - len(accepted_rois))
            while len(pending) < pending_limit and attempt_idx <= max_attempts:
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
                    static_context=env_static,
                )
                roi, _actions, info = rollout_one_slice(policy, env, device_t)
                repeat_attempt_overlap = _attempt_overlap(roi, attempted_rois)
                attempted_rois.append(roi)
                if _skip_crop_reason(info, cfg) is not None:
                    consecutive_rejections += 1
                    if repeat_attempt_overlap >= 0.95 or (
                        rejection_limit > 0 and consecutive_rejections >= rejection_limit
                    ):
                        stop_attempts = True
                        break
                else:
                    pending.append(roi)
                attempt_idx += 1
            if not pending:
                continue
            predictions = run_yolo_on_crops(
                crop_model,
                [image_path] * len(pending),
                pending,
                imgsz=cfg.slice_imgsz,
                conf=cfg.output_conf,
                iou=cfg.iou,
                max_det=cfg.max_det,
                device=cfg.device,
                source_image=source_image,
            )
            before = len(accepted_rois)
            for roi, prediction in zip(pending, predictions):
                accept_prediction(roi, prediction)
            consecutive_rejections = 0 if len(accepted_rois) > before else consecutive_rejections + len(pending)
            if rejection_limit > 0 and consecutive_rejections >= rejection_limit:
                stop_attempts = True

    return {
        "full": (full_boxes, full_scores, full_classes),
        "accepted": (accepted_boxes, accepted_scores, accepted_classes),
        "candidates": (candidate_boxes, candidate_scores, candidate_classes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle analysis for RL-SAHI localization and merge ceiling.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "benchmark" / "oracle_analysis")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--sampling", choices=("stratified", "sequential"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    project_cfg = load_default_config(args.config, ROOT)
    infer_cfg, bench_cfg = _build_configs(
        project_cfg,
        args.policy_device or project_cfg.optional_str("infer", "policy_device") or project_cfg.optional_str("infer", "device"),
    )
    if args.sampling is not None:
        bench_cfg.sampling = args.sampling
    if args.seed is not None:
        bench_cfg.seed = int(args.seed)

    checkpoint = project_cfg.path_value("checkpoint") if args.checkpoint is None else args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir

    print_device_info("oracle-detector", infer_cfg.device)
    print_device_info("oracle-policy", infer_cfg.policy_device)
    print(
        f"[oracle] split={args.split} limit={args.limit} output_conf={infer_cfg.output_conf:g} "
        f"merge_iou={infer_cfg.merge_iou:g} nms_type={infer_cfg.nms_type}"
    )

    image_root = project_cfg.path_value("image_root")
    label_root = project_cfg.path_value("label_root")
    annotation_root = _path_or_none(project_cfg, "annotation_root")
    cache_root = project_cfg.path_value("cache_root")
    weights = project_cfg.path_value("weights")
    full_weights = _path_or_none(project_cfg, "full_weights")
    crop_weights = _path_or_none(project_cfg, "crop_weights")
    images = select_benchmark_images(
        iter_images(image_root, split=args.split),
        args.limit,
        sampling=bench_cfg.sampling,
        seed=bench_cfg.seed,
    )
    if not images:
        raise FileNotFoundError(f"No images found for split={args.split!r}")

    small_threshold = _resolve_small_area_threshold(images, image_root, label_root, bench_cfg, annotation_root)
    detector_device_t = resolve_torch_device(infer_cfg.device)
    model = load_yolo(weights, device=detector_device_t)
    full_model = load_yolo(full_weights, device=detector_device_t) if full_weights else model
    crop_model = load_yolo(crop_weights, device=detector_device_t) if crop_weights else model
    device_t = resolve_torch_device(infer_cfg.policy_device or infer_cfg.device)
    policy, checkpoint_data = load_policy(checkpoint, device_t)
    env_cfg = checkpoint_data["env_cfg_obj"]
    state_cfg = checkpoint_data.get("state_cfg_obj", StateConfig())
    use_cache = _bool_value(project_cfg.section("infer").get("use_cache", True)) and not args.no_cache

    ground_truth: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray]] = {}
    predictions: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        "normal_merge": {},
        "oracle_cluster_accepted": {},
        "oracle_per_gt_accepted": {},
        "oracle_cluster_all_candidates": {},
        "oracle_per_gt_all_candidates": {},
    }
    recall_totals = {
        "accepted_candidates": {threshold: [0, 0] for threshold in AP_THRESHOLDS},
        "all_candidates": {threshold: [0, 0] for threshold in AP_THRESHOLDS},
    }
    rows_per_image: list[dict[str, float | str]] = []
    start_all = time.perf_counter()

    for image_index, image_path in enumerate(images, start=1):
        if image_index == 1 or image_index % 25 == 0 or image_index == len(images):
            print(f"[oracle] image {image_index}/{len(images)} {image_path.name}", flush=True)
        image_id = image_path.stem
        gt_boxes, gt_classes, ignore_boxes = _read_gt(
            image_path,
            image_root,
            label_root,
            bench_cfg.target_classes,
            bench_cfg.class_mapping,
            annotation_root,
            bench_cfg.ignore_overlap_threshold,
        )
        source_image = read_image(image_path)
        shape = (int(source_image.shape[0]), int(source_image.shape[1]))
        ground_truth[image_id] = (gt_boxes, gt_classes, shape, ignore_boxes)
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
            split=args.split,
            use_cache=use_cache,
            source_image=source_image,
        )
        parts = _collect_parts(
            model,
            full_model,
            crop_model,
            policy,
            device_t,
            image_path,
            det,
            infer_cfg,
            env_cfg,
            state_cfg,
            source_image,
        )
        full_boxes, full_scores, full_classes = parts["full"]  # type: ignore[assignment]
        accepted_boxes, accepted_scores, accepted_classes = parts["accepted"]  # type: ignore[assignment]
        candidate_boxes, candidate_scores, candidate_classes = parts["candidates"]  # type: ignore[assignment]

        accepted_boxes_parts = [full_boxes, *accepted_boxes]
        accepted_scores_parts = [full_scores, *accepted_scores]
        accepted_classes_parts = [full_classes, *accepted_classes]
        all_boxes_parts = [full_boxes, *candidate_boxes]
        all_scores_parts = [full_scores, *candidate_scores]
        all_classes_parts = [full_classes, *candidate_classes]

        normal_boxes, normal_scores, normal_classes, _sources = _merge_from_parts(
            shape,
            infer_cfg,
            accepted_boxes_parts,
            accepted_scores_parts,
            accepted_classes_parts,
        )
        predictions["normal_merge"][image_id] = (normal_boxes, normal_scores, normal_classes)

        accepted_all_boxes = _concat_parts(accepted_boxes_parts, (0, 4), np.float32)
        accepted_all_scores = _concat_parts(accepted_scores_parts, (0,), np.float32)
        accepted_all_classes = _concat_parts(accepted_classes_parts, (0,), np.float32)
        candidate_all_boxes = _concat_parts(all_boxes_parts, (0, 4), np.float32)
        candidate_all_scores = _concat_parts(all_scores_parts, (0,), np.float32)
        candidate_all_classes = _concat_parts(all_classes_parts, (0,), np.float32)

        predictions["oracle_cluster_accepted"][image_id] = _oracle_cluster_predictions(
            shape,
            accepted_all_boxes,
            accepted_all_scores,
            accepted_all_classes,
            gt_boxes,
            gt_classes,
            infer_cfg.merge_iou,
        )
        predictions["oracle_per_gt_accepted"][image_id] = _oracle_per_gt_predictions(
            accepted_all_boxes,
            accepted_all_scores,
            accepted_all_classes,
            gt_boxes,
            gt_classes,
        )
        predictions["oracle_cluster_all_candidates"][image_id] = _oracle_cluster_predictions(
            shape,
            candidate_all_boxes,
            candidate_all_scores,
            candidate_all_classes,
            gt_boxes,
            gt_classes,
            infer_cfg.merge_iou,
        )
        predictions["oracle_per_gt_all_candidates"][image_id] = _oracle_per_gt_predictions(
            candidate_all_boxes,
            candidate_all_scores,
            candidate_all_classes,
            gt_boxes,
            gt_classes,
        )
        _candidate_recall(
            accepted_all_boxes,
            accepted_all_classes,
            ground_truth,
            image_id,
            recall_totals["accepted_candidates"],
        )
        _candidate_recall(
            candidate_all_boxes,
            candidate_all_classes,
            ground_truth,
            image_id,
            recall_totals["all_candidates"],
        )
        rows_per_image.append(
            {
                "image_id": image_id,
                "gt": float(len(gt_boxes)),
                "full_boxes": float(len(full_boxes)),
                "accepted_crop_boxes": float(sum(len(x) for x in accepted_boxes)),
                "candidate_crop_boxes": float(sum(len(x) for x in candidate_boxes)),
                "accepted_slices": float(len(accepted_boxes)),
                "candidate_slices": float(len(candidate_boxes)),
                "normal_boxes": float(len(normal_boxes)),
            }
        )

    metric_rows: list[dict[str, float | str]] = []
    for name, method_predictions in predictions.items():
        metrics = _evaluate_method(
            method_predictions,
            ground_truth,
            bench_cfg.target_classes,
            bench_cfg.iou_threshold,
            small_threshold,
            bench_cfg.eval_max_detections,
            bench_cfg.ignore_overlap_threshold,
        )
        row = {"method": name, **metrics}
        metric_rows.append(row)
        print(
            f"[oracle] {name}: AP={metrics['AP']:.4f} AP75={metrics['AP75']:.4f} "
            f"AP90={metrics['AP90']:.4f} AP95={metrics['AP95']:.4f} "
            f"recall={metrics['recall']:.4f} fp/image={metrics['fp_per_image']:.2f}",
            flush=True,
        )

    candidate_recall = {
        name: {
            f"R{int(threshold * 100):02d}": float(hit / max(total, 1))
            for threshold, (hit, total) in totals.items()
        }
        for name, totals in recall_totals.items()
    }
    for name, values in candidate_recall.items():
        print(
            f"[oracle] {name}: R75={values['R75']:.4f} "
            f"R90={values['R90']:.4f} R95={values['R95']:.4f}",
            flush=True,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    metric_path = out_dir / f"{args.split}_limit{len(images)}_metrics.csv"
    with metric_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)
    image_path_out = out_dir / f"{args.split}_limit{len(images)}_per_image.csv"
    with image_path_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_per_image[0].keys()))
        writer.writeheader()
        writer.writerows(rows_per_image)
    summary_path = out_dir / f"{args.split}_limit{len(images)}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "images": len(images),
                "elapsed_s": time.perf_counter() - start_all,
                "weights": file_fingerprint(weights),
                "full_weights": file_fingerprint(full_weights) if full_weights else None,
                "crop_weights": file_fingerprint(crop_weights) if crop_weights else None,
                "checkpoint": file_fingerprint(checkpoint),
                "inference_config": asdict(infer_cfg),
                "benchmark_config": asdict(bench_cfg),
                "metrics": metric_rows,
                "candidate_recall": candidate_recall,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[oracle] wrote {metric_path}")
    print(f"[oracle] wrote {summary_path}")


if __name__ == "__main__":
    main()
