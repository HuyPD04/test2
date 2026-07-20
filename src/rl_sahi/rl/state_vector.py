from __future__ import annotations

import numpy as np


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    if feature.size == 0:
        return feature
    feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(feature))
    if norm > 1e-6:
        feature = feature / norm
    return np.clip(feature, -5.0, 5.0).astype(np.float32)


def build_state_vector(
    feature: np.ndarray,
    history: np.ndarray,
    current_roi_map: np.ndarray,
    attempted_slice_map: np.ndarray,
    accepted_slice_map: np.ndarray,
    detection_map: np.ndarray,
    objectness_map: np.ndarray,
    spatial_feature_map: np.ndarray,
    summary: np.ndarray,
    static_ready: bool = False,
    out_buffer: np.ndarray | None = None,
) -> np.ndarray:
    if static_ready:
        feature_part = np.asarray(feature, dtype=np.float32)
        objectness = np.asarray(objectness_map, dtype=np.float32)
        spatial = np.asarray(spatial_feature_map, dtype=np.float32)
    else:
        feature_part = normalize_feature(feature)
        objectness = np.nan_to_num(
            np.asarray(objectness_map, dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        spatial = np.nan_to_num(
            np.asarray(spatial_feature_map, dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        
    if out_buffer is None:
        total_size = (
            feature_part.size + history.size + current_roi_map.size +
            attempted_slice_map.size + accepted_slice_map.size +
            detection_map.size + objectness.size + spatial.size + summary.size
        )
        out_buffer = np.zeros(total_size, dtype=np.float32)

    offset = 0
    for arr in (feature_part, history, current_roi_map, attempted_slice_map, 
                accepted_slice_map, detection_map, objectness, spatial, summary):
        s = arr.size
        out_buffer[offset : offset + s] = np.asarray(arr, dtype=np.float32).ravel()
        offset += s

    return out_buffer
