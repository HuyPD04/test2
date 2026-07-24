from __future__ import annotations

import numpy as np

from ..config import AnchorConfig, EnvironmentConfig
from .geometry import iou_matrix, union_area_ratio
from .types import Anchor, Detections


ANCHOR_FEATURES = 12
GLOBAL_FEATURES = 8


def state_dimension(anchor_cfg: AnchorConfig) -> int:
    return anchor_cfg.top_k * ANCHOR_FEATURES + GLOBAL_FEATURES


def build_state(
    anchors: list[Anchor],
    full_detections: Detections,
    attempted_rois: list[np.ndarray],
    accepted_rois: list[np.ndarray],
    selected_anchors: set[int],
    image_shape: tuple[int, int],
    last_utility: float,
    anchor_cfg: AnchorConfig,
    env_cfg: EnvironmentConfig,
) -> np.ndarray:
    height, width = image_shape
    image_area = float(max(height * width, 1))
    attempted = np.asarray(attempted_rois, dtype=np.float32).reshape(-1, 4)
    rows = np.zeros((anchor_cfg.top_k, ANCHOR_FEATURES), dtype=np.float32)
    for index, anchor in enumerate(anchors[: anchor_cfg.top_k]):
        anchor_box = np.asarray(
            [[
                anchor.cx - anchor.width / 2.0,
                anchor.cy - anchor.height / 2.0,
                anchor.cx + anchor.width / 2.0,
                anchor.cy + anchor.height / 2.0,
            ]],
            dtype=np.float32,
        )
        history_iou = float(iou_matrix(anchor_box, attempted).max()) if len(attempted) else 0.0
        rows[index] = np.asarray(
            [
                anchor.cx / max(width, 1),
                anchor.cy / max(height, 1),
                anchor.width / max(width, 1),
                anchor.height / max(height, 1),
                anchor.score,
                min(anchor.count / max(anchor_cfg.density_norm, 1.0), 1.0),
                anchor.mean_conf,
                anchor.max_conf,
                anchor.small_fraction,
                min(anchor.mean_area_ratio / max(anchor_cfg.small_area_ratio, 1e-7), 1.0),
                history_iou,
                1.0 if index in selected_anchors else 0.0,
            ],
            dtype=np.float32,
        )

    full_areas = (
        (full_detections.boxes[:, 2] - full_detections.boxes[:, 0])
        * (full_detections.boxes[:, 3] - full_detections.boxes[:, 1])
        / image_area
        if len(full_detections)
        else np.zeros((0,), dtype=np.float32)
    )
    low_conf_fraction = (
        float(np.mean(full_detections.scores < anchor_cfg.low_confidence)) if len(full_detections) else 0.0
    )
    small_fraction = (
        float(np.mean(full_areas <= anchor_cfg.small_area_ratio)) if len(full_areas) else 0.0
    )
    global_features = np.asarray(
        [
            1.0 - len(attempted_rois) / max(env_cfg.max_crops, 1),
            len(attempted_rois) / max(env_cfg.max_crops, 1),
            len(accepted_rois) / max(env_cfg.max_crops, 1),
            min(len(full_detections) / max(anchor_cfg.full_count_norm, 1.0), 1.0),
            low_conf_fraction,
            small_fraction,
            union_area_ratio(attempted_rois, image_shape),
            min(max(float(last_utility), 0.0), 1.0),
        ],
        dtype=np.float32,
    )
    return np.concatenate([rows.reshape(-1), global_features]).astype(np.float32)
