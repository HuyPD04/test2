from __future__ import annotations

import numpy as np

from rl_sahi.common.box_geometry import iou_matrix
from rl_sahi.common.box_types import as_boxes


def weighted_box_fusion(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    classes_list: list[np.ndarray],
    iou_threshold: float = 0.5,
    skip_box_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if not boxes_list:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    all_boxes = np.concatenate(
        [as_boxes(b) for b in boxes_list], axis=0
    ) if any(len(b) > 0 for b in boxes_list) else np.zeros((0, 4), dtype=np.float32)
    all_scores = np.concatenate(
        [np.asarray(s, dtype=np.float32).reshape(-1) for s in scores_list], axis=0
    ) if any(len(s) > 0 for s in scores_list) else np.zeros((0,), dtype=np.float32)
    all_classes = np.concatenate(
        [np.asarray(c, dtype=np.float32).reshape(-1) for c in classes_list], axis=0
    ) if any(len(c) > 0 for c in classes_list) else np.zeros((0,), dtype=np.float32)

    if len(all_boxes) == 0:
        return all_boxes, all_scores, all_classes

    # Filter by skip_box_threshold
    if skip_box_threshold > 0:
        keep = all_scores >= skip_box_threshold
        all_boxes = all_boxes[keep]
        all_scores = all_scores[keep]
        all_classes = all_classes[keep]
        if len(all_boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

    fused_boxes: list[np.ndarray] = []
    fused_scores: list[float] = []
    fused_classes: list[float] = []

    for cls in np.unique(all_classes.astype(np.int64)):
        cls_mask = all_classes.astype(np.int64) == cls
        cls_boxes = all_boxes[cls_mask]
        cls_scores = all_scores[cls_mask]

        if len(cls_boxes) == 0:
            continue

        # Sort by score descending
        order = np.argsort(cls_scores)[::-1]
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]

        # Clusters: each cluster is a list of (box, score) that will be fused
        clusters: list[list[int]] = []  # indices into cls_boxes
        cluster_boxes: list[np.ndarray] = []  # current fused box for each cluster
        cluster_scores: list[float] = []  # current fused score for each cluster

        for idx in range(len(cls_boxes)):
            box = cls_boxes[idx]
            score = float(cls_scores[idx])
            matched_cluster = -1

            if cluster_boxes:
                # Compute IoU with all existing cluster representative boxes
                ious = iou_matrix(
                    box.reshape(1, 4),
                    np.stack(cluster_boxes).astype(np.float32),
                )[0]
                best_cluster = int(ious.argmax())
                if float(ious[best_cluster]) >= iou_threshold:
                    matched_cluster = best_cluster

            if matched_cluster >= 0:
                # Add to existing cluster and re-fuse
                clusters[matched_cluster].append(idx)
                members = clusters[matched_cluster]
                member_boxes = cls_boxes[members]
                member_scores = cls_scores[members]
                weights = member_scores / max(float(member_scores.sum()), 1e-9)
                fused = (member_boxes * weights[:, None]).sum(axis=0)
                cluster_boxes[matched_cluster] = fused
                cluster_scores[matched_cluster] = float(member_scores.max())
            else:
                # Create new cluster
                clusters.append([idx])
                cluster_boxes.append(box.copy())
                cluster_scores.append(score)

        for i in range(len(clusters)):
            fused_boxes.append(cluster_boxes[i])
            fused_scores.append(cluster_scores[i])
            fused_classes.append(float(cls))

    if not fused_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    out_boxes = np.stack(fused_boxes).astype(np.float32)
    out_scores = np.asarray(fused_scores, dtype=np.float32)
    out_classes = np.asarray(fused_classes, dtype=np.float32)

    # Sort by score descending
    order = np.argsort(out_scores)[::-1]
    return out_boxes[order], out_scores[order], out_classes[order]
