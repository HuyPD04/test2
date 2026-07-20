from __future__ import annotations

import numpy as np

from rl_sahi.common.boxes import area, as_boxes, intersection_matrix, normalized_box
from rl_sahi.rl.state_config import SUMMARY_DIM, StateConfig
from rl_sahi.rl.state_maps import proposal_mask, proposal_quality


def detection_summary(
    roi: np.ndarray,
    history: np.ndarray,
    previous_slice_map: np.ndarray,
    image_shape: tuple[int, int],
    step_index: int,
    max_steps: int,
    old_slice_overlap: float,
    scale_gain: float,
    previous_slice_count: int,
    cfg: StateConfig,
    # cached properties:
    boxes: np.ndarray,
    scores: np.ndarray,
    summary_static: np.ndarray,
    boxes_area: np.ndarray,
    prop_mask: np.ndarray,
    prop_quality: np.ndarray,
    small_mask: np.ndarray,
    low_mask: np.ndarray,
) -> np.ndarray:
    summary = summary_static.copy()
    image_area = float(image_shape[0] * image_shape[1])

    if len(boxes) > 0:
        inter = intersection_matrix(np.asarray(roi, dtype=np.float32).reshape(1, 4), boxes)[0]
        in_roi = inter > 0.0
        if in_roi.any():
            roi_scores = scores[in_roi]
            roi_areas = boxes_area[in_roi]
            roi_prop_mask = prop_mask[in_roi]
            roi_prop_quality = prop_quality[in_roi]
            summary[6] = min(float(in_roi.sum()) / cfg.roi_count_norm, 1.0)
            summary[7] = float(roi_scores.mean())
            summary[8] = float(roi_scores.max())
            summary[9] = min(float(low_mask[in_roi].sum()) / cfg.roi_count_norm, 1.0)
            summary[10] = min(float(small_mask[in_roi].sum()) / cfg.roi_count_norm, 1.0)
            roi_area = max(float(area(np.asarray(roi).reshape(1, 4))[0]), 1.0)
            summary[11] = float(np.clip(inter[in_roi].sum() / roi_area, 0.0, 1.0))
            summary[25] = min(float(roi_prop_mask.sum()) / cfg.roi_count_norm, 1.0)
            if roi_prop_mask.any():
                summary[27] = float(roi_prop_quality[roi_prop_mask].mean())

    nb = normalized_box(roi, image_shape)
    summary[12] = (nb[0] + nb[2]) / 2.0
    summary[13] = (nb[1] + nb[3]) / 2.0
    summary[14] = nb[2] - nb[0]
    summary[15] = nb[3] - nb[1]
    summary[16] = float(np.clip(area(np.asarray(roi).reshape(1, 4))[0] / max(image_area, 1.0), 0.0, 1.0))
    summary[17] = float(step_index / max(max_steps, 1))
    summary[18] = float(np.clip(history.mean(), 0.0, 1.0))
    summary[19] = float(np.clip((image_shape[1] / max(image_shape[0], 1)) / 4.0, 0.0, 1.0))
    summary[20] = float(np.clip(np.asarray(previous_slice_map, dtype=np.float32).mean(), 0.0, 1.0))
    summary[21] = float(np.clip(old_slice_overlap, 0.0, 1.0))
    summary[22] = float(np.clip(scale_gain / 8.0, 0.0, 1.0))
    summary[23] = min(float(previous_slice_count) / cfg.slice_count_norm, 1.0)
    return summary
