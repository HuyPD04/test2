"""Greedy Non-Maximum Merging (NMM) — the SAHI default post-processing.

Unlike NMS which *suppresses* overlapping boxes (keeping only the highest
score), Greedy NMM *merges* them by averaging coordinates weighted by
confidence.  This is critical for sliced inference where the same object
may be partially detected in adjacent overlapping slices.

The algorithm also supports **IOS** (Intersection over Smaller area) as the
match metric instead of the standard IoU, which SAHI uses by default.  IOS
is better for small-object scenarios because a small box fully inside a
larger one still receives a high match score.

Reference: Akyon et al., "Slicing Aided Hyper Inference and Fine-tuning
for Small Object Detection", ICIP 2022.
"""

from __future__ import annotations

import numpy as np

from rl_sahi.common.box_geometry import area, ios_matrix, iou_matrix
from rl_sahi.common.box_types import as_boxes


def _match_matrix(
    boxes: np.ndarray,
    match_metric: str,
) -> np.ndarray:
    """Compute the pairwise match matrix using the requested metric."""
    if match_metric == "IOS":
        return ios_matrix(boxes, boxes)
    return iou_matrix(boxes, boxes)


def greedy_nmm(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    match_metric: str = "IOS",
    match_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Class-aware Greedy Non-Maximum Merging.

    For each class independently:
      1. Sort detections by confidence (descending).
      2. Take the highest-confidence box as a cluster seed.
      3. Greedily absorb all remaining boxes whose match score with the
         seed exceeds *match_threshold*.
      4. Merge the cluster into a single box whose coordinates are the
         confidence-weighted average of its members.  The merged score
         is the maximum confidence in the cluster (following the SAHI
         library convention).
      5. Repeat until no boxes remain.

    Args:
        boxes:  (N, 4) xyxy boxes.
        scores: (N,) confidence scores.
        classes: (N,) class IDs.
        match_metric: ``"IOU"`` or ``"IOS"`` (default).
        match_threshold: Minimum overlap to merge (default 0.5).

    Returns:
        Tuple of (merged_boxes, merged_scores, merged_classes), sorted
        by descending score.
    """
    boxes = as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    classes = np.asarray(classes, dtype=np.float32).reshape(-1)

    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    merged_boxes: list[np.ndarray] = []
    merged_scores: list[float] = []
    merged_classes: list[float] = []

    for cls in np.unique(classes.astype(np.int64)):
        cls_mask = classes.astype(np.int64) == cls
        cls_boxes = boxes[cls_mask].copy()
        cls_scores = scores[cls_mask].copy()

        if len(cls_boxes) == 0:
            continue

        # Pre-compute pairwise match matrix for this class
        match_mat = _match_matrix(cls_boxes, match_metric)

        # Sort by score descending
        order = np.argsort(cls_scores)[::-1]
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]
        match_mat = match_mat[np.ix_(order, order)]

        remaining = np.ones(len(cls_boxes), dtype=bool)

        for i in range(len(cls_boxes)):
            if not remaining[i]:
                continue

            # Find all remaining boxes that match with box i
            matches = remaining.copy()
            matches[i] = True  # always include seed
            for j in range(i + 1, len(cls_boxes)):
                if remaining[j] and match_mat[i, j] >= match_threshold:
                    matches[j] = True

            # Collect cluster members
            cluster_indices = np.where(matches)[0]
            cluster_boxes = cls_boxes[cluster_indices]
            cluster_scores = cls_scores[cluster_indices]

            # Mark all cluster members as consumed
            remaining[cluster_indices] = False

            # Merge: weighted average of coordinates by confidence
            if len(cluster_indices) == 1:
                merged_box = cluster_boxes[0]
            else:
                weights = cluster_scores / max(float(cluster_scores.sum()), 1e-9)
                merged_box = (cluster_boxes * weights[:, None]).sum(axis=0)

            merged_boxes.append(merged_box.astype(np.float32))
            merged_scores.append(float(cluster_scores.max()))
            merged_classes.append(float(cls))

    if not merged_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    out_boxes = np.stack(merged_boxes).astype(np.float32)
    out_scores = np.asarray(merged_scores, dtype=np.float32)
    out_classes = np.asarray(merged_classes, dtype=np.float32)

    # Sort by score descending
    order = np.argsort(out_scores)[::-1]
    return out_boxes[order], out_scores[order], out_classes[order]
