from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from rl_sahi.common.boxes import area, clip_boxes, intersection_matrix
from rl_sahi.common.cache import (
    DetectionCache,
    detection_cache_is_current,
    detection_cache_metadata,
    detection_cache_path,
    file_fingerprint,
    load_detection_cache,
    save_detection_cache,
)
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import ProjectConfig, load_default_config
from rl_sahi.common.device import DeviceLike, resolve_torch_device
from rl_sahi.detection.yolo import detect_one_image, load_yolo
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.crops import run_yolo_on_crops
from rl_sahi.inference.merge import (
    accepts_novel_detections,
    class_aware_nms,
    new_detection_gain_after_merge,
    new_detection_stats_after_merge,
    new_detection_utility_after_merge,
    save_prediction_txt,
    source_counts_after_merge,
)
from rl_sahi.inference.rollout import rollout_one_slice
from rl_sahi.inference.visualize import save_inference_visual
from rl_sahi.rl.checkpoint import load_policy
from rl_sahi.rl.slice_env import SliceEnv
from rl_sahi.rl.state_config import StateConfig


def _class_mask(classes: np.ndarray, target_classes: tuple[int, ...]) -> np.ndarray:
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    if not target_classes:
        return np.ones((len(classes),), dtype=bool)
    return np.isin(classes.astype(np.int64), np.asarray(target_classes, dtype=np.int64))


def _filter_classes(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    target_classes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = _class_mask(classes, target_classes)
    return boxes[mask], scores[mask], classes[mask]


def _merged_source_counts(
    full_boxes: np.ndarray,
    full_scores: np.ndarray,
    full_classes: np.ndarray,
    slice_boxes_parts: list[np.ndarray],
    slice_scores_parts: list[np.ndarray],
    slice_classes_parts: list[np.ndarray],
    image_shape: tuple[int, int],
    merge_iou: float,
) -> tuple[int, int]:
    return source_counts_after_merge(
        full_boxes,
        full_scores,
        full_classes,
        slice_boxes_parts,
        slice_scores_parts,
        slice_classes_parts,
        image_shape,
        merge_iou,
    )


def _new_detection_gain(
    full_boxes: np.ndarray,
    full_scores: np.ndarray,
    full_classes: np.ndarray,
    slice_boxes_parts: list[np.ndarray],
    slice_scores_parts: list[np.ndarray],
    slice_classes_parts: list[np.ndarray],
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
    image_shape: tuple[int, int],
    merge_iou: float,
    duplicate_iou: float | None = None,
) -> int:
    return new_detection_gain_after_merge(
        image_shape,
        merge_iou,
        [full_boxes, *slice_boxes_parts],
        [full_scores, *slice_scores_parts],
        [full_classes, *slice_classes_parts],
        candidate_boxes,
        candidate_scores,
        candidate_classes,
        duplicate_iou=duplicate_iou,
    )


def _new_detection_utility(
    full_boxes: np.ndarray,
    full_scores: np.ndarray,
    full_classes: np.ndarray,
    slice_boxes_parts: list[np.ndarray],
    slice_scores_parts: list[np.ndarray],
    slice_classes_parts: list[np.ndarray],
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
    image_shape: tuple[int, int],
    merge_iou: float,
    duplicate_iou: float | None = None,
) -> float:
    return new_detection_utility_after_merge(
        image_shape,
        merge_iou,
        [full_boxes, *slice_boxes_parts],
        [full_scores, *slice_scores_parts],
        [full_classes, *slice_classes_parts],
        candidate_boxes,
        candidate_scores,
        candidate_classes,
        duplicate_iou=duplicate_iou,
    )


def _new_detection_stats(
    full_boxes: np.ndarray,
    full_scores: np.ndarray,
    full_classes: np.ndarray,
    slice_boxes_parts: list[np.ndarray],
    slice_scores_parts: list[np.ndarray],
    slice_classes_parts: list[np.ndarray],
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
    image_shape: tuple[int, int],
    merge_iou: float,
    duplicate_iou: float | None = None,
) -> tuple[int, float, float]:
    return new_detection_stats_after_merge(
        image_shape,
        merge_iou,
        [full_boxes, *slice_boxes_parts],
        [full_scores, *slice_scores_parts],
        [full_classes, *slice_classes_parts],
        candidate_boxes,
        candidate_scores,
        candidate_classes,
        duplicate_iou=duplicate_iou,
    )


def _accept_slice(gain: int, utility: float, max_score: float, cfg: InferenceConfig) -> bool:
    return accepts_novel_detections(
        gain,
        utility,
        max_score,
        cfg.min_slice_detections,
        cfg.min_slice_utility,
        cfg.min_new_detection_score,
    )


def _crop_rejection_reason(
    raw_detection_count: int,
    gain: int,
    utility: float,
    max_score: float,
    cfg: InferenceConfig,
) -> str | None:
    if _accept_slice(gain, utility, max_score, cfg):
        return None
    if raw_detection_count == 0:
        return "empty_slice"
    if gain <= 0:
        return "no_new_detection_after_nms"
    if max_score < float(cfg.min_new_detection_score):
        return "low_new_detection_score"
    if gain > 1 and utility < float(cfg.min_slice_utility):
        return "low_new_detection_utility"
    return "low_new_detection_count"


def _attempt_overlap(roi: np.ndarray, attempted_rois: list[np.ndarray]) -> float:
    if not attempted_rois:
        return 0.0
    previous = np.stack(attempted_rois).astype(np.float32)
    roi_arr = np.asarray(roi, dtype=np.float32).reshape(1, 4)
    inter = intersection_matrix(roi_arr, previous)[0]
    current_area = max(float(area(roi_arr)[0]), 1.0)
    return float(np.clip(inter.max() / current_area, 0.0, 1.0))


def _skip_crop_reason(info: dict, cfg: InferenceConfig) -> str | None:
    if info.get("stop_due_to_old_overlap", False):
        return "old_slice_overlap"
    if info.get("stop_due_to_attempted_overlap", False):
        return "attempted_slice_overlap"
    if cfg.require_stop_for_acceptance and info.get("stop_due_to_max_steps", False):
        return "max_steps_without_stop"
    if cfg.require_stop_for_acceptance and info.get("stop_due_to_stalled_roi", False):
        return "stalled_without_stop"
    return None


def _checkpoint_detection_mismatches(
    metadata: dict | None,
    cfg: InferenceConfig,
    state_cfg: StateConfig,
    weights: Path,
) -> list[str]:
    if not isinstance(metadata, dict) or not metadata:
        return []
    expected = {
        "imgsz": int(cfg.full_imgsz),
        "conf": float(cfg.full_conf),
        "iou": float(cfg.iou),
        "max_det": int(cfg.max_det),
        "feature_layers": tuple(int(x) for x in cfg.feature_layers),
        "aux_grid_size": int(state_cfg.grid_size),
        "spatial_feature_channels": int(state_cfg.spatial_feature_channels),
    }
    mismatches: list[str] = []
    actual_weights = metadata.get("weights")
    if isinstance(actual_weights, dict):
        expected_weights = file_fingerprint(weights)
        if actual_weights.get("sha256") and expected_weights.get("sha256"):
            if actual_weights["sha256"] != expected_weights["sha256"]:
                mismatches.append("weights SHA-256 differs from the training checkpoint")
        elif actual_weights.get("size") != expected_weights.get("size"):
            mismatches.append(
                f"weights size: checkpoint={actual_weights.get('size')!r}, "
                f"inference={expected_weights.get('size')!r}"
            )
    for key, expected_value in expected.items():
        if key not in metadata:
            continue
        actual_value = metadata[key]
        if key == "feature_layers":
            actual_value = tuple(int(x) for x in actual_value)
        elif isinstance(expected_value, int):
            actual_value = int(actual_value)
        elif isinstance(expected_value, float):
            actual_value = float(actual_value)
        if actual_value != expected_value:
            mismatches.append(f"{key}: checkpoint={actual_value!r}, inference={expected_value!r}")
    return mismatches


def _checkpoint_semantic_mismatches(metadata: dict | None, cfg: InferenceConfig) -> list[str]:
    if not isinstance(metadata, dict) or not metadata:
        return []
    mismatches: list[str] = []
    checkpoint_targets = metadata.get("target_classes")
    if checkpoint_targets is not None:
        actual = tuple(int(x) for x in checkpoint_targets)
        if actual != tuple(int(x) for x in cfg.target_classes):
            mismatches.append(
                f"target_classes: checkpoint={actual!r}, inference={cfg.target_classes!r}"
            )
    checkpoint_mapping = metadata.get("class_mapping")
    expected_mapping = {
        "model_to_label": dict(cfg.class_mapping.model_to_label),
        "label_to_eval": dict(cfg.class_mapping.label_to_eval),
    }
    if isinstance(checkpoint_mapping, dict) and checkpoint_mapping != expected_mapping:
        mismatches.append("class_mapping differs from the training checkpoint")
    return mismatches


def get_initial_detection(
    model: YOLO,
    weights: Path | None,
    image_path: Path,
    weights_imgsz: int,
    full_conf: float,
    full_iou: float,
    max_det: int,
    device: DeviceLike,
    feature_layers: tuple[int, ...],
    aux_grid_size: int,
    spatial_feature_channels: int,
    cache_root: Path | str | None = None,
    split: str | None = None,
    use_cache: bool = True,
) -> DetectionCache:
    expected_metadata = (
        detection_cache_metadata(
            weights=weights,
            imgsz=weights_imgsz,
            conf=full_conf,
            iou=full_iou,
            max_det=max_det,
            feature_layers=feature_layers,
            aux_grid_size=aux_grid_size,
            spatial_feature_channels=spatial_feature_channels,
        )
        if weights is not None
        else None
    )
    if cache_root is not None and split is not None:
        cache_path = detection_cache_path(cache_root, split, image_path)
        if use_cache and detection_cache_is_current(cache_path, expected_metadata):
            return load_detection_cache(cache_path)
        det = detect_one_image(
            model=model,
            image_path=image_path,
            imgsz=weights_imgsz,
            conf=full_conf,
            iou=full_iou,
            max_det=max_det,
            device=device,
            feature_layers=feature_layers,
            aux_grid_size=aux_grid_size,
            spatial_feature_channels=spatial_feature_channels,
        )
        det.metadata = expected_metadata
        save_detection_cache(cache_path, det)
        return det
    det = detect_one_image(
        model=model,
        image_path=image_path,
        imgsz=weights_imgsz,
        conf=full_conf,
        iou=full_iou,
        max_det=max_det,
        device=device,
        feature_layers=feature_layers,
        aux_grid_size=aux_grid_size,
        spatial_feature_channels=spatial_feature_channels,
    )
    det.metadata = expected_metadata
    return det


class AdaptiveSahiInferencer:
    def __init__(self, weights: Path, checkpoint: Path, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        self.device_t = resolve_torch_device(cfg.device)
        self.policy, checkpoint_data = load_policy(checkpoint, self.device_t)
        self.env_cfg = checkpoint_data["env_cfg_obj"]
        self.state_cfg = checkpoint_data.get("state_cfg_obj", StateConfig())
        mismatches = _checkpoint_detection_mismatches(
            checkpoint_data.get("detection_metadata"),
            cfg,
            self.state_cfg,
            Path(weights),
        )
        mismatches.extend(
            _checkpoint_semantic_mismatches(checkpoint_data.get("training_metadata"), cfg)
        )
        if mismatches:
            raise ValueError(
                "Checkpoint detection metadata does not match inference config: "
                + "; ".join(mismatches)
            )
        self.weights = Path(weights)
        self.provenance = {
            "weights": file_fingerprint(Path(weights)),
            "checkpoint": file_fingerprint(Path(checkpoint)),
            "inference_config": asdict(cfg),
        }
        self.yolo = load_yolo(weights, device=self.device_t)

    def infer_image(
        self,
        image_path: Path,
        out_dir: Path,
        cache_root: Path | None = None,
        split: str | None = None,
        use_cache: bool = True,
    ) -> dict:
        cfg = self.cfg
        request_start = time.perf_counter()
        detection_start = time.perf_counter()
        det = get_initial_detection(
            model=self.yolo,
            weights=self.weights,
            image_path=image_path,
            weights_imgsz=cfg.full_imgsz,
            full_conf=cfg.full_conf,
            full_iou=cfg.iou,
            max_det=cfg.max_det,
            device=cfg.device,
            feature_layers=cfg.feature_layers,
            aux_grid_size=self.state_cfg.grid_size,
            spatial_feature_channels=self.state_cfg.spatial_feature_channels,
            cache_root=cache_root,
            split=split,
            use_cache=use_cache,
        )
        initial_detection_ms = (time.perf_counter() - detection_start) * 1000.0

        return _infer_with_loaded(
            image_path=image_path,
            out_dir=out_dir,
            yolo=self.yolo,
            policy=self.policy,
            device_t=self.device_t,
            env_cfg=self.env_cfg,
            state_cfg=self.state_cfg,
            det=det,
            cfg=cfg,
            initial_detection_ms=initial_detection_ms,
            request_start=request_start,
            provenance=self.provenance,
        )


def _infer_with_loaded(
    image_path: Path,
    out_dir: Path,
    yolo: YOLO,
    policy,
    device_t: torch.device,
    env_cfg,
    state_cfg: StateConfig,
    det: DetectionCache,
    cfg: InferenceConfig,
    initial_detection_ms: float = 0.0,
    request_start: float | None = None,
    provenance: dict | None = None,
) -> dict:
    if request_start is None:
        request_start = time.perf_counter()
    timing = {
        "initial_detection_ms": float(initial_detection_ms),
        "rollout_ms": 0.0,
        "crop_inference_ms": 0.0,
        "merge_ms": 0.0,
        "write_outputs_ms": 0.0,
        "total_ms": 0.0,
    }
    accepted_rois: list[np.ndarray] = []
    rejected_rois: list[np.ndarray] = []
    attempted_rois: list[np.ndarray] = []
    slice_boxes_all: list[np.ndarray] = []
    slice_scores_all: list[np.ndarray] = []
    slice_classes_all: list[np.ndarray] = []
    slice_meta: list[dict] = []

    full_mask = det.scores >= cfg.output_conf
    full_boxes = det.boxes[full_mask]
    full_scores = det.scores[full_mask]
    full_classes = cfg.class_mapping.map_model_classes(det.classes[full_mask])
    full_boxes, full_scores, full_classes = _filter_classes(
        full_boxes,
        full_scores,
        full_classes,
        cfg.target_classes,
    )
    max_attempts = int(cfg.max_slice_attempts) if cfg.max_slice_attempts > 0 else int(env_cfg.max_slices * 2)
    crop_batch_size = max(int(cfg.crop_batch_size), 1)
    crop_prediction_count = 0
    crop_batch_count = 0
    global_stop_reason: str | None = None

    if cfg.batched_inference:
        # ── BATCHED MODE: 3-phase pipeline ──────────────────────────
        # Phase 1: Collect all ROI candidates (NO YOLO, geometry-only gate)
        # Phase 2: Batch YOLO on all candidates at once
        # Phase 3: Greedy post-filter with gain/utility
        candidate_rois: list[tuple[int, np.ndarray, list[str], dict]] = []
        attempt_idx = 1

        rollout_start = time.perf_counter()
        while attempt_idx <= max_attempts and len(candidate_rois) < env_cfg.max_slices:
            history_arr = (
                np.stack(attempted_rois).astype(np.float32)
                if attempted_rois
                else np.zeros((0, 4), dtype=np.float32)
            )
            # In batched mode, use attempted_rois as overlap check too,
            # since we don't know accepted rois until after YOLO runs.
            env = SliceEnv(
                det,
                None,
                env_cfg=env_cfg,
                state_cfg=state_cfg,
                previous_rois=history_arr,
                overlap_rois=history_arr,
                target_classes=cfg.target_classes,
                class_mapping=cfg.class_mapping,
            )
            roi, actions, info = rollout_one_slice(policy, env, device_t)
            repeat_attempt_overlap = _attempt_overlap(roi, attempted_rois)
            attempted_rois.append(roi)
            skip_reason = _skip_crop_reason(info, cfg)
            if skip_reason is not None:
                rejected_rois.append(roi)
                slice_meta.append(
                    {
                        "attempt_index": attempt_idx,
                        "slice_index": None,
                        "accepted": False,
                        "rejection_reason": skip_reason,
                        "roi": [float(x) for x in roi.tolist()],
                        "actions": actions,
                        "steps": len(actions),
                        "old_slice_overlap": float(info.get("old_slice_overlap", 0.0)),
                        "attempted_slice_overlap": float(info.get("attempted_slice_overlap", 0.0)),
                        "stop_due_to_stalled_roi": bool(info.get("stop_due_to_stalled_roi", False)),
                        "repeat_attempt_overlap": repeat_attempt_overlap,
                        "detections": 0,
                    }
                )
                if repeat_attempt_overlap >= 0.95:
                    break
            else:
                candidate_rois.append((attempt_idx, roi, actions, info))
            attempt_idx += 1
        timing["rollout_ms"] = (time.perf_counter() - rollout_start) * 1000.0

        # Phase 2: Batch YOLO on all candidates at once
        if candidate_rois:
            crop_start = time.perf_counter()
            crop_predictions = run_yolo_on_crops(
                yolo,
                [image_path] * len(candidate_rois),
                [roi for _, roi, _, _ in candidate_rois],
                imgsz=cfg.slice_imgsz,
                conf=cfg.output_conf,
                iou=cfg.iou,
                max_det=cfg.max_det,
                device=cfg.device,
            )
            timing["crop_inference_ms"] = (time.perf_counter() - crop_start) * 1000.0
            crop_prediction_count = len(candidate_rois)
            crop_batch_count = 1

            # Phase 3: Greedy post-filter — accept candidates by gain/utility
            for (cand_attempt_idx, roi, actions, info), (boxes_i, scores_i, classes_i) in zip(
                candidate_rois,
                crop_predictions,
            ):
                classes_i = cfg.class_mapping.map_model_classes(classes_i)
                boxes_i, scores_i, classes_i = _filter_classes(boxes_i, scores_i, classes_i, cfg.target_classes)
                new_detection_gain, new_detection_utility, new_detection_max_score = _new_detection_stats(
                    full_boxes, full_scores, full_classes,
                    slice_boxes_all, slice_scores_all, slice_classes_all,
                    boxes_i, scores_i, classes_i,
                    det.image_shape, cfg.merge_iou, cfg.duplicate_iou,
                )
                rejection_reason = _crop_rejection_reason(
                    len(boxes_i), new_detection_gain, new_detection_utility,
                    new_detection_max_score, cfg,
                )
                accepted = rejection_reason is None
                slice_index = None
                if accepted:
                    accepted_rois.append(roi)
                    slice_index = len(accepted_rois)
                    slice_boxes_all.append(boxes_i)
                    slice_scores_all.append(scores_i)
                    slice_classes_all.append(classes_i)
                else:
                    rejected_rois.append(roi)
                slice_meta.append(
                    {
                        "attempt_index": cand_attempt_idx,
                        "slice_index": slice_index,
                        "accepted": bool(accepted),
                        "rejection_reason": rejection_reason,
                        "roi": [float(x) for x in roi.tolist()],
                        "actions": actions,
                        "steps": len(actions),
                        "old_slice_overlap": float(info.get("old_slice_overlap", 0.0)),
                        "attempted_slice_overlap": float(info.get("attempted_slice_overlap", 0.0)),
                        "stop_due_to_stalled_roi": bool(info.get("stop_due_to_stalled_roi", False)),
                        "detections": int(len(boxes_i)),
                        "new_detections_after_nms": int(new_detection_gain),
                        "new_detection_utility": float(new_detection_utility),
                        "new_detection_max_score": float(new_detection_max_score),
                        "crop_batch_size": len(candidate_rois),
                        "batched": True,
                    }
                )
    else:
        # ── SEQUENTIAL MODE (original) ──────────────────────────────
        attempt_idx = 1
        stop_attempts = False
        consecutive_rejections = 0
        rejection_limit = max(int(cfg.max_consecutive_rejections), 0)

        while attempt_idx <= max_attempts and len(accepted_rois) < env_cfg.max_slices and not stop_attempts:
            remaining_attempts = max_attempts - attempt_idx + 1
            remaining_slices = max(int(env_cfg.max_slices) - len(accepted_rois), 1)
            pending_limit = min(crop_batch_size, remaining_attempts, remaining_slices)
            pending: list[tuple[int, np.ndarray, list[str], dict]] = []

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
                rollout_start = time.perf_counter()
                roi, actions, info = rollout_one_slice(policy, env, device_t)
                timing["rollout_ms"] += (time.perf_counter() - rollout_start) * 1000.0
                repeat_attempt_overlap = _attempt_overlap(roi, attempted_rois)
                attempted_rois.append(roi)
                skip_reason = _skip_crop_reason(info, cfg)
                if skip_reason is not None:
                    consecutive_rejections += 1
                    rejected_rois.append(roi)
                    slice_meta.append(
                        {
                            "attempt_index": attempt_idx,
                            "slice_index": None,
                            "accepted": False,
                            "rejection_reason": skip_reason,
                            "roi": [float(x) for x in roi.tolist()],
                            "actions": actions,
                            "steps": len(actions),
                            "old_slice_overlap": float(info.get("old_slice_overlap", 0.0)),
                            "attempted_slice_overlap": float(info.get("attempted_slice_overlap", 0.0)),
                            "stop_due_to_stalled_roi": bool(info.get("stop_due_to_stalled_roi", False)),
                            "repeat_attempt_overlap": repeat_attempt_overlap,
                            "detections": 0,
                        }
                    )
                    if repeat_attempt_overlap >= 0.95:
                        stop_attempts = True
                        global_stop_reason = "repeat_attempt"
                        break
                    if rejection_limit > 0 and consecutive_rejections >= rejection_limit:
                        stop_attempts = True
                        global_stop_reason = "consecutive_rejections"
                        break
                else:
                    pending.append((attempt_idx, roi, actions, info))
                attempt_idx += 1

            if not pending:
                continue

            crop_start = time.perf_counter()
            crop_predictions = run_yolo_on_crops(
                yolo,
                [image_path] * len(pending),
                [roi for _pending_attempt_idx, roi, _actions, _info in pending],
                imgsz=cfg.slice_imgsz,
                conf=cfg.output_conf,
                iou=cfg.iou,
                max_det=cfg.max_det,
                device=cfg.device,
            )
            timing["crop_inference_ms"] += (time.perf_counter() - crop_start) * 1000.0
            crop_prediction_count += len(pending)
            crop_batch_count += 1

            for (pending_attempt_idx, roi, actions, info), (boxes_i, scores_i, classes_i) in zip(
                pending,
                crop_predictions,
            ):
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
                rejection_reason = _crop_rejection_reason(
                    len(boxes_i), new_detection_gain, new_detection_utility,
                    new_detection_max_score, cfg,
                )
                accepted = rejection_reason is None
                slice_index = None
                if accepted:
                    consecutive_rejections = 0
                    accepted_rois.append(roi)
                    slice_index = len(accepted_rois)
                    slice_boxes_all.append(boxes_i)
                    slice_scores_all.append(scores_i)
                    slice_classes_all.append(classes_i)
                else:
                    consecutive_rejections += 1
                    rejected_rois.append(roi)
                slice_meta.append(
                    {
                        "attempt_index": pending_attempt_idx,
                        "slice_index": slice_index,
                        "accepted": bool(accepted),
                        "rejection_reason": rejection_reason,
                        "roi": [float(x) for x in roi.tolist()],
                        "actions": actions,
                        "steps": len(actions),
                        "old_slice_overlap": float(info.get("old_slice_overlap", 0.0)),
                        "attempted_slice_overlap": float(info.get("attempted_slice_overlap", 0.0)),
                        "stop_due_to_stalled_roi": bool(info.get("stop_due_to_stalled_roi", False)),
                        "detections": int(len(boxes_i)),
                        "new_detections_after_nms": int(new_detection_gain),
                        "new_detection_utility": float(new_detection_utility),
                        "new_detection_max_score": float(new_detection_max_score),
                        "crop_batch_size": len(pending),
                    }
                )
            if rejection_limit > 0 and consecutive_rejections >= rejection_limit:
                stop_attempts = True
                global_stop_reason = "consecutive_rejections"

    merge_start = time.perf_counter()
    boxes_parts = [full_boxes] + slice_boxes_all
    scores_parts = [full_scores] + slice_scores_all
    classes_parts = [full_classes] + slice_classes_all
    sources_parts = [np.zeros((len(full_boxes),), dtype=np.int32)] + [
        np.full((len(boxes_i),), index + 1, dtype=np.int32)
        for index, boxes_i in enumerate(slice_boxes_all)
    ]

    boxes = np.concatenate(boxes_parts, axis=0) if boxes_parts else np.zeros((0, 4), dtype=np.float32)
    scores = np.concatenate(scores_parts, axis=0) if scores_parts else np.zeros((0,), dtype=np.float32)
    classes = np.concatenate(classes_parts, axis=0) if classes_parts else np.zeros((0,), dtype=np.float32)
    sources = np.concatenate(sources_parts, axis=0) if sources_parts else np.zeros((0,), dtype=np.int32)

    boxes = clip_boxes(boxes, det.image_shape)
    keep = class_aware_nms(boxes, scores, classes, cfg.merge_iou)
    boxes, scores, classes, sources = boxes[keep], scores[keep], classes[keep], sources[keep]
    timing["merge_ms"] = (time.perf_counter() - merge_start) * 1000.0

    out_dir = Path(out_dir)
    pred_path = out_dir / "detections" / f"{image_path.stem}.txt"
    viz_path = out_dir / "visualizations" / f"{image_path.stem}.jpg"
    meta_path = out_dir / "metadata" / f"{image_path.stem}.json"
    accepted_rois_array = (
        np.stack(accepted_rois).astype(np.float32) if accepted_rois else np.zeros((0, 4), dtype=np.float32)
    )
    rejected_rois_array = (
        np.stack(rejected_rois).astype(np.float32) if rejected_rois else np.zeros((0, 4), dtype=np.float32)
    )
    write_start = time.perf_counter()
    if cfg.save_predictions:
        save_prediction_txt(pred_path, boxes, scores, classes, sources)
    if cfg.save_visualization:
        save_inference_visual(image_path, boxes, sources, accepted_rois_array, rejected_rois_array, viz_path)
    timing["write_outputs_ms"] = (time.perf_counter() - write_start) * 1000.0
    timing["total_ms"] = (time.perf_counter() - request_start) * 1000.0
    meta = {
        "image": str(image_path),
        "num_slices": len(accepted_rois),
        "num_attempts": len(slice_meta),
        "num_rejected_slices": len(rejected_rois),
        "num_crop_predictions": crop_prediction_count,
        "num_crop_batches": crop_batch_count,
        "slices": slice_meta,
        "detections": int(len(boxes)),
        "prediction_file": str(pred_path) if cfg.save_predictions else None,
        "visualization_file": str(viz_path) if cfg.save_visualization else None,
        "metadata_file": str(meta_path) if cfg.save_metadata else None,
        "timing": timing,
        "batched_inference": bool(cfg.batched_inference),
        "global_stop_reason": global_stop_reason,
        "provenance": provenance or {},
    }
    if cfg.save_metadata:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _resolve_project_path(path: Path | str, root: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else root / value


def _config_path_or_override(cfg: ProjectConfig, key: str, value: Path | str | None) -> Path:
    if value is None:
        return cfg.path_value(key)
    return _resolve_project_path(value, cfg.root)


def _value_or_config(section: dict, key: str, value, cast):
    raw = section[key] if value is None else value
    return cast(raw)


def _optional_value_or_config(section: dict, key: str, value, default, cast):
    raw = section.get(key, default) if value is None else value
    return cast(raw)


def _bool_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _feature_layers_or_config(cfg: ProjectConfig, value: tuple[int, ...] | list[int] | str | None) -> tuple[int, ...]:
    if value is None:
        return cfg.feature_layers("infer")
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def _int_tuple_value(value: tuple[int, ...] | list[int] | str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def infer_one_image(
    image_path: Path | str,
    weights: Path | str | None = None,
    checkpoint: Path | str | None = None,
    out_dir: Path | str | None = None,
    cache_root: Path | None = None,
    split: str | None = None,
    use_cache: bool | None = None,
    full_imgsz: int | None = None,
    slice_imgsz: int | None = None,
    full_conf: float | None = None,
    output_conf: float | None = None,
    iou: float | None = None,
    merge_iou: float | None = None,
    max_det: int | None = None,
    device: str | None = None,
    feature_layers: tuple[int, ...] | list[int] | str | None = None,
    min_slice_detections: int | None = None,
    min_slice_utility: float | None = None,
    min_new_detection_score: float | None = None,
    duplicate_iou: float | None = None,
    max_slice_attempts: int | None = None,
    crop_batch_size: int | None = None,
    max_consecutive_rejections: int | None = None,
    target_classes: tuple[int, ...] | list[int] | str | None = None,
    require_stop_for_acceptance: bool | None = None,
    save_predictions: bool | None = None,
    save_metadata: bool | None = None,
    save_visualization: bool | None = None,
    batched_inference: bool | None = None,
    class_mapping: ClassMapping | None = None,
    config: ProjectConfig | Path | str | None = None,
) -> dict:
    project_cfg = config if isinstance(config, ProjectConfig) else load_default_config(config)
    infer_cfg = project_cfg.section("infer")
    image_path = _resolve_project_path(image_path, project_cfg.root)
    weights = _config_path_or_override(project_cfg, "weights", weights)
    checkpoint = _config_path_or_override(project_cfg, "checkpoint", checkpoint)
    out_dir = _config_path_or_override(project_cfg, "infer_out_dir", out_dir)
    cache_root = _config_path_or_override(project_cfg, "cache_root", cache_root)
    use_cache = bool(infer_cfg.get("use_cache", True)) if use_cache is None else bool(use_cache)

    cfg = InferenceConfig(
        full_imgsz=_value_or_config(infer_cfg, "full_imgsz", full_imgsz, int),
        slice_imgsz=_value_or_config(infer_cfg, "slice_imgsz", slice_imgsz, int),
        full_conf=_value_or_config(infer_cfg, "full_conf", full_conf, float),
        output_conf=_value_or_config(infer_cfg, "output_conf", output_conf, float),
        iou=_value_or_config(infer_cfg, "iou", iou, float),
        merge_iou=_value_or_config(infer_cfg, "merge_iou", merge_iou, float),
        max_det=_value_or_config(infer_cfg, "max_det", max_det, int),
        device=device if device is not None else project_cfg.optional_str("infer", "device"),
        feature_layers=_feature_layers_or_config(project_cfg, feature_layers),
        min_slice_detections=_value_or_config(infer_cfg, "min_slice_detections", min_slice_detections, int),
        min_slice_utility=(
            float(infer_cfg.get("min_slice_utility", 0.5))
            if min_slice_utility is None
            else float(min_slice_utility)
        ),
        min_new_detection_score=(
            float(infer_cfg.get("min_new_detection_score", 0.45))
            if min_new_detection_score is None
            else float(min_new_detection_score)
        ),
        duplicate_iou=(
            float(infer_cfg.get("duplicate_iou", infer_cfg.get("merge_iou", 0.5)))
            if duplicate_iou is None
            else float(duplicate_iou)
        ),
        max_slice_attempts=_value_or_config(infer_cfg, "max_slice_attempts", max_slice_attempts, int),
        crop_batch_size=_optional_value_or_config(infer_cfg, "crop_batch_size", crop_batch_size, 1, int),
        max_consecutive_rejections=_optional_value_or_config(
            infer_cfg, "max_consecutive_rejections", max_consecutive_rejections, 0, int
        ),
        target_classes=_int_tuple_value(
            target_classes if target_classes is not None else project_cfg.target_classes()
        ),
        require_stop_for_acceptance=(
            _bool_value(infer_cfg.get("require_stop_for_acceptance", True))
            if require_stop_for_acceptance is None
            else _bool_value(require_stop_for_acceptance)
        ),
        save_predictions=(
            _bool_value(infer_cfg.get("save_predictions", True))
            if save_predictions is None
            else _bool_value(save_predictions)
        ),
        save_metadata=(
            _bool_value(infer_cfg.get("save_metadata", True))
            if save_metadata is None
            else _bool_value(save_metadata)
        ),
        save_visualization=(
            _bool_value(infer_cfg.get("save_visualization", False))
            if save_visualization is None
            else _bool_value(save_visualization)
        ),
        batched_inference=(
            _bool_value(infer_cfg.get("batched_inference", False))
            if batched_inference is None
            else _bool_value(batched_inference)
        ),
        class_mapping=class_mapping or ClassMapping.from_config(project_cfg.section("classes")),
    )
    inferencer = AdaptiveSahiInferencer(weights=weights, checkpoint=checkpoint, cfg=cfg)
    return inferencer.infer_image(
        image_path=image_path,
        out_dir=out_dir,
        cache_root=cache_root,
        split=split,
        use_cache=use_cache,
    )
