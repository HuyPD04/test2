from __future__ import annotations

from pathlib import Path

import numpy as np

from rl_sahi.common.boxes import as_boxes, clip_boxes, iou_matrix, nms_numpy
from rl_sahi.common.wbf import weighted_box_fusion


DEFAULT_VEHICLE_DUPLICATE_CLASSES = (3, 4, 5, 8)  # car, van, truck, bus
DEFAULT_CROSS_CLASS_DUPLICATE_GROUPS = (DEFAULT_VEHICLE_DUPLICATE_CLASSES,)
DEFAULT_CROSS_CLASS_DUPLICATE_IOU = 0.85
DEFAULT_CROSS_CLASS_DUPLICATE_IOS = 0.95
DEFAULT_SOURCE_AWARE_SLICE_BONUS = 0.05


def save_prediction_txt(
    path: Path,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    sources: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cls, score, box, source in zip(classes, scores, boxes, sources):
            x1, y1, x2, y2 = [float(v) for v in box]
            f.write(f"{int(cls)} {float(score):.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {int(source)}\n")


def _split_nms_type(nms_type: str) -> tuple[bool, str]:
    name = str(nms_type).lower().replace("-", "_")
    if name in {"source_aware", "sourceaware"}:
        return True, "standard"
    if name.startswith("source_aware_"):
        return True, name.removeprefix("source_aware_") or "standard"
    if name.endswith("_source_aware"):
        return True, name.removesuffix("_source_aware") or "standard"
    return False, name


def _source_priority_scores(
    scores: np.ndarray,
    sources: np.ndarray,
    slice_bonus: float = DEFAULT_SOURCE_AWARE_SLICE_BONUS,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    sources = np.asarray(sources, dtype=np.int32).reshape(-1)
    return scores + (sources > 0).astype(np.float32) * float(slice_bonus)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
    nms_type: str = "standard",
) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    _source_aware, base_nms_type = _split_nms_type(nms_type)
    keep_parts: list[np.ndarray] = []
    for cls in np.unique(classes.astype(np.int64)):
        idx = np.flatnonzero(classes.astype(np.int64) == cls)
        keep_local = nms_numpy(boxes[idx], scores[idx], iou_threshold, nms_type=base_nms_type)
        keep_parts.append(idx[keep_local])
    keep = np.concatenate(keep_parts, axis=0) if keep_parts else np.zeros((0,), dtype=np.int64)
    return keep[np.argsort(scores[keep])[::-1]].astype(np.int64)


def class_aware_nms_with_sources(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    sources: np.ndarray,
    iou_threshold: float,
    nms_type: str = "standard",
) -> np.ndarray:
    boxes = as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    sources = np.asarray(sources, dtype=np.int32).reshape(-1)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    if not (len(boxes) == len(scores) == len(classes) == len(sources)):
        raise ValueError("boxes, scores, classes, and sources must have the same length")

    source_aware, base_nms_type = _split_nms_type(nms_type)
    if not source_aware:
        return class_aware_nms(boxes, scores, classes, iou_threshold, nms_type=base_nms_type)

    keep_parts: list[np.ndarray] = []
    for cls in np.unique(classes.astype(np.int64)):
        idx = np.flatnonzero(classes.astype(np.int64) == cls)
        priority_scores = _source_priority_scores(scores[idx], sources[idx])
        keep_local = nms_numpy(boxes[idx], priority_scores, iou_threshold, nms_type=base_nms_type)
        keep_parts.append(idx[keep_local])
    keep = np.concatenate(keep_parts, axis=0) if keep_parts else np.zeros((0,), dtype=np.int64)
    return keep[np.argsort(scores[keep])[::-1]].astype(np.int64)


def cross_class_duplicate_keep(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOU,
    ios_threshold: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOS,
    class_groups: tuple[tuple[int, ...], ...] = DEFAULT_CROSS_CLASS_DUPLICATE_GROUPS,
) -> np.ndarray:
    boxes = as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes_i = np.asarray(classes, dtype=np.float32).reshape(-1).astype(np.int64)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)
    if (
        (iou_threshold is None or float(iou_threshold) <= 0.0)
        and (ios_threshold is None or float(ios_threshold) <= 0.0)
    ):
        return np.arange(len(boxes), dtype=np.int64)

    iou_threshold = None if iou_threshold is None else float(np.clip(float(iou_threshold), 0.0, 1.0))
    ios_threshold = None if ios_threshold is None else float(np.clip(float(ios_threshold), 0.0, 1.0))
    areas = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 0.0, None)
    keep_mask = np.ones((len(boxes),), dtype=bool)
    for group in class_groups:
        group_classes = np.asarray(tuple(int(cls) for cls in group), dtype=np.int64)
        if len(group_classes) < 2:
            continue
        group_idx = np.flatnonzero(np.isin(classes_i, group_classes))
        if len(group_idx) < 2:
            continue
        order = group_idx[np.argsort(scores[group_idx])[::-1]]
        for pos, current in enumerate(order):
            if not keep_mask[current]:
                continue
            rest = order[pos + 1 :]
            rest = rest[keep_mask[rest] & (classes_i[rest] != classes_i[current])]
            if len(rest) == 0:
                continue
            suppress = np.zeros((len(rest),), dtype=bool)
            if iou_threshold is not None and iou_threshold > 0.0:
                ious = iou_matrix(boxes[current].reshape(1, 4), boxes[rest])[0]
                suppress |= ious >= iou_threshold
            if ios_threshold is not None and ios_threshold > 0.0:
                inter_x1 = np.maximum(boxes[current, 0], boxes[rest, 0])
                inter_y1 = np.maximum(boxes[current, 1], boxes[rest, 1])
                inter_x2 = np.minimum(boxes[current, 2], boxes[rest, 2])
                inter_y2 = np.minimum(boxes[current, 3], boxes[rest, 3])
                inter = np.clip(inter_x2 - inter_x1, 0.0, None) * np.clip(inter_y2 - inter_y1, 0.0, None)
                smaller_area = np.maximum(np.minimum(areas[current], areas[rest]), 1e-9)
                suppress |= (inter / smaller_area) >= ios_threshold
            keep_mask[rest[suppress]] = False
    keep = np.flatnonzero(keep_mask).astype(np.int64)
    return keep[np.argsort(scores[keep])[::-1]].astype(np.int64)


def resolve_cross_class_duplicates(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOU,
    ios_threshold: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    keep = cross_class_duplicate_keep(boxes, scores, classes, iou_threshold, ios_threshold)
    return boxes[keep], scores[keep], classes[keep]


def merge_predictions(
    image_shape: tuple[int, int],
    merge_iou: float,
    boxes_parts: list[np.ndarray],
    scores_parts: list[np.ndarray],
    classes_parts: list[np.ndarray],
    use_wbf: bool = False,
    nms_type: str = "standard",
    cross_class_duplicate_iou: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOU,
    cross_class_duplicate_ios: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.concatenate(boxes_parts, axis=0) if boxes_parts else np.zeros((0, 4), dtype=np.float32)
    scores = np.concatenate(scores_parts, axis=0) if scores_parts else np.zeros((0,), dtype=np.float32)
    classes = np.concatenate(classes_parts, axis=0) if classes_parts else np.zeros((0,), dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return boxes, scores, classes
    boxes = clip_boxes(boxes, image_shape)
    if use_wbf:
        boxes, scores, classes = weighted_box_fusion([boxes], [scores], [classes], iou_threshold=merge_iou)
    else:
        source_aware, _base_nms_type = _split_nms_type(nms_type)
        if source_aware:
            sources = np.zeros((len(boxes),), dtype=np.int32)
            keep = class_aware_nms_with_sources(boxes, scores, classes, sources, merge_iou, nms_type=nms_type)
        else:
            keep = class_aware_nms(boxes, scores, classes, merge_iou, nms_type=nms_type)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    return resolve_cross_class_duplicates(boxes, scores, classes, cross_class_duplicate_iou, cross_class_duplicate_ios)


def merge_predictions_with_sources(
    image_shape: tuple[int, int],
    merge_iou: float,
    boxes_parts: list[np.ndarray],
    scores_parts: list[np.ndarray],
    classes_parts: list[np.ndarray],
    sources_parts: list[np.ndarray],
    cross_class_duplicate_iou: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOU,
    cross_class_duplicate_ios: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOS,
    use_wbf: bool = False,
    nms_type: str = "standard",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.concatenate(boxes_parts, axis=0) if boxes_parts else np.zeros((0, 4), dtype=np.float32)
    scores = np.concatenate(scores_parts, axis=0) if scores_parts else np.zeros((0,), dtype=np.float32)
    classes = np.concatenate(classes_parts, axis=0) if classes_parts else np.zeros((0,), dtype=np.float32)
    sources = np.concatenate(sources_parts, axis=0) if sources_parts else np.zeros((0,), dtype=np.int32)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    sources = np.asarray(sources, dtype=np.int32).reshape(-1)
    if len(boxes) == 0:
        return boxes, scores, classes, sources
    boxes = clip_boxes(boxes, image_shape)
    if use_wbf:
        original_boxes = boxes.copy()
        original_sources = sources.copy()
        boxes, scores, classes = weighted_box_fusion([boxes], [scores], [classes], iou_threshold=merge_iou)
        sources = _assign_fused_sources(boxes, original_boxes, original_sources)
    else:
        keep = class_aware_nms_with_sources(boxes, scores, classes, sources, merge_iou, nms_type=nms_type)
        boxes, scores, classes, sources = boxes[keep], scores[keep], classes[keep], sources[keep]
    keep = cross_class_duplicate_keep(boxes, scores, classes, cross_class_duplicate_iou, cross_class_duplicate_ios)
    return boxes[keep], scores[keep], classes[keep], sources[keep]


def _assign_fused_sources(
    fused_boxes: np.ndarray,
    original_boxes: np.ndarray,
    original_sources: np.ndarray,
) -> np.ndarray:
    fused_boxes = as_boxes(fused_boxes)
    original_boxes = as_boxes(original_boxes)
    original_sources = np.asarray(original_sources, dtype=np.int32).reshape(-1)
    if len(fused_boxes) == 0 or len(original_boxes) == 0:
        return np.zeros((len(fused_boxes),), dtype=np.int32)
    match_ious = iou_matrix(fused_boxes, original_boxes)
    return original_sources[match_ious.argmax(axis=1)]


def source_counts_after_merge(
    full_boxes: np.ndarray,
    full_scores: np.ndarray,
    full_classes: np.ndarray,
    slice_boxes_parts: list[np.ndarray],
    slice_scores_parts: list[np.ndarray],
    slice_classes_parts: list[np.ndarray],
    image_shape: tuple[int, int],
    merge_iou: float,
    cross_class_duplicate_iou: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOU,
    cross_class_duplicate_ios: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOS,
    use_wbf: bool = False,
    nms_type: str = "standard",
) -> tuple[int, int]:
    sources_parts = [np.zeros((len(full_boxes),), dtype=np.int32)] + [
        np.full((len(boxes),), index + 1, dtype=np.int32)
        for index, boxes in enumerate(slice_boxes_parts)
    ]
    _boxes, _scores, _classes, sources = merge_predictions_with_sources(
        image_shape,
        merge_iou,
        [full_boxes, *slice_boxes_parts],
        [full_scores, *slice_scores_parts],
        [full_classes, *slice_classes_parts],
        sources_parts,
        cross_class_duplicate_iou=cross_class_duplicate_iou,
        cross_class_duplicate_ios=cross_class_duplicate_ios,
        use_wbf=use_wbf,
        nms_type=nms_type,
    )
    return int((sources == 0).sum()), int((sources > 0).sum())


def _novel_candidate_detections_after_merge(
    image_shape: tuple[int, int],
    merge_iou: float,
    previous_boxes_parts: list[np.ndarray],
    previous_scores_parts: list[np.ndarray],
    previous_classes_parts: list[np.ndarray],
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
    duplicate_iou: float | None = None,
    cross_class_duplicate_iou: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOU,
    cross_class_duplicate_ios: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOS,
    use_wbf: bool = False,
    nms_type: str = "standard",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before_boxes, _before_scores, before_classes = merge_predictions(
        image_shape,
        merge_iou,
        previous_boxes_parts,
        previous_scores_parts,
        previous_classes_parts,
        use_wbf=use_wbf,
        nms_type=nms_type,
        cross_class_duplicate_iou=cross_class_duplicate_iou,
        cross_class_duplicate_ios=cross_class_duplicate_ios,
    )
    candidate_boxes = np.asarray(candidate_boxes, dtype=np.float32).reshape(-1, 4)
    candidate_scores = np.asarray(candidate_scores, dtype=np.float32).reshape(-1)
    candidate_classes = np.asarray(candidate_classes, dtype=np.float32).reshape(-1)
    if len(candidate_boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    sources_parts = [
        np.zeros((sum(len(boxes) for boxes in previous_boxes_parts),), dtype=np.int32),
        np.ones((len(candidate_boxes),), dtype=np.int32),
    ]
    after_boxes, after_scores, after_classes, after_sources = merge_predictions_with_sources(
        image_shape,
        merge_iou,
        [*previous_boxes_parts, candidate_boxes],
        [*previous_scores_parts, candidate_scores],
        [*previous_classes_parts, candidate_classes],
        sources_parts,
        cross_class_duplicate_iou=cross_class_duplicate_iou,
        cross_class_duplicate_ios=cross_class_duplicate_ios,
        use_wbf=use_wbf,
        nms_type=nms_type,
    )
    candidate_mask = after_sources == 1
    if not candidate_mask.any():
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    if len(before_boxes) == 0:
        return after_boxes[candidate_mask], after_scores[candidate_mask], after_classes[candidate_mask]

    duplicate_threshold = float(merge_iou) if duplicate_iou is None else float(duplicate_iou)
    duplicate_threshold = float(np.clip(duplicate_threshold, 0.0, 1.0))
    novel = np.ones((int(candidate_mask.sum()),), dtype=bool)
    for idx, (box, cls) in enumerate(zip(after_boxes[candidate_mask], after_classes[candidate_mask])):
        same_cls = before_classes.astype(np.int64) == int(cls)
        if not same_cls.any():
            continue
        ious = iou_matrix(box.reshape(1, 4), before_boxes[same_cls])[0]
        if len(ious) > 0 and float(ious.max()) >= duplicate_threshold:
            novel[idx] = False
    return (
        after_boxes[candidate_mask][novel],
        after_scores[candidate_mask][novel],
        after_classes[candidate_mask][novel],
    )


def new_detection_gain_after_merge(
    image_shape: tuple[int, int],
    merge_iou: float,
    previous_boxes_parts: list[np.ndarray],
    previous_scores_parts: list[np.ndarray],
    previous_classes_parts: list[np.ndarray],
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
    duplicate_iou: float | None = None,
    cross_class_duplicate_iou: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOU,
    cross_class_duplicate_ios: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOS,
    use_wbf: bool = False,
    nms_type: str = "standard",
) -> int:
    gain, _utility, _max_score = new_detection_stats_after_merge(
        image_shape,
        merge_iou,
        previous_boxes_parts,
        previous_scores_parts,
        previous_classes_parts,
        candidate_boxes,
        candidate_scores,
        candidate_classes,
        duplicate_iou=duplicate_iou,
        cross_class_duplicate_iou=cross_class_duplicate_iou,
        cross_class_duplicate_ios=cross_class_duplicate_ios,
        use_wbf=use_wbf,
        nms_type=nms_type,
    )
    return gain


def new_detection_utility_after_merge(
    image_shape: tuple[int, int],
    merge_iou: float,
    previous_boxes_parts: list[np.ndarray],
    previous_scores_parts: list[np.ndarray],
    previous_classes_parts: list[np.ndarray],
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
    duplicate_iou: float | None = None,
    cross_class_duplicate_iou: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOU,
    cross_class_duplicate_ios: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOS,
    use_wbf: bool = False,
    nms_type: str = "standard",
) -> float:
    _gain, utility, _max_score = new_detection_stats_after_merge(
        image_shape,
        merge_iou,
        previous_boxes_parts,
        previous_scores_parts,
        previous_classes_parts,
        candidate_boxes,
        candidate_scores,
        candidate_classes,
        duplicate_iou=duplicate_iou,
        cross_class_duplicate_iou=cross_class_duplicate_iou,
        cross_class_duplicate_ios=cross_class_duplicate_ios,
        use_wbf=use_wbf,
        nms_type=nms_type,
    )
    return utility


def new_detection_stats_after_merge(
    image_shape: tuple[int, int],
    merge_iou: float,
    previous_boxes_parts: list[np.ndarray],
    previous_scores_parts: list[np.ndarray],
    previous_classes_parts: list[np.ndarray],
    candidate_boxes: np.ndarray,
    candidate_scores: np.ndarray,
    candidate_classes: np.ndarray,
    duplicate_iou: float | None = None,
    cross_class_duplicate_iou: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOU,
    cross_class_duplicate_ios: float | None = DEFAULT_CROSS_CLASS_DUPLICATE_IOS,
    use_wbf: bool = False,
    nms_type: str = "standard",
) -> tuple[int, float, float]:
    boxes, scores, _classes = _novel_candidate_detections_after_merge(
        image_shape,
        merge_iou,
        previous_boxes_parts,
        previous_scores_parts,
        previous_classes_parts,
        candidate_boxes,
        candidate_scores,
        candidate_classes,
        duplicate_iou=duplicate_iou,
        cross_class_duplicate_iou=cross_class_duplicate_iou,
        cross_class_duplicate_ios=cross_class_duplicate_ios,
        use_wbf=use_wbf,
        nms_type=nms_type,
    )
    if len(boxes) == 0:
        return 0, 0.0, 0.0
    clipped_scores = np.clip(scores, 0.0, 1.0)
    return int(len(boxes)), float(clipped_scores.sum()), float(clipped_scores.max())


def accepts_novel_detections(
    gain: int,
    utility: float,
    max_score: float,
    min_detections: int,
    min_utility: float,
    min_score: float,
) -> bool:
    if int(gain) < int(min_detections) or float(max_score) < float(min_score):
        return False
    return int(gain) == 1 or float(utility) >= float(min_utility)
