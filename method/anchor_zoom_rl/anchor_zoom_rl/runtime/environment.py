from __future__ import annotations

import numpy as np

from ..config import AnchorConfig, EnvironmentConfig
from ..core.geometry import anchor_to_roi, iou_matrix
from ..core.state import build_state
from ..core.types import Anchor, Detections


class AnchorZoomEnvironment:
    def __init__(
        self,
        anchors: list[Anchor],
        full_detections: Detections,
        image_shape: tuple[int, int],
        anchor_cfg: AnchorConfig,
        env_cfg: EnvironmentConfig,
    ) -> None:
        self.anchors = anchors
        self.full_detections = full_detections
        self.image_shape = image_shape
        self.anchor_cfg = anchor_cfg
        self.env_cfg = env_cfg
        self.attempted_rois: list[np.ndarray] = []
        self.accepted_rois: list[np.ndarray] = []
        self.selected_anchors: set[int] = set()
        self.last_utility = 0.0

    @property
    def action_count(self) -> int:
        return self.anchor_cfg.top_k * len(self.anchor_cfg.zoom_bins) + 1

    @property
    def stop_action(self) -> int:
        return self.action_count - 1

    @property
    def done(self) -> bool:
        return len(self.attempted_rois) >= self.env_cfg.max_crops

    def decode_action(self, action: int) -> tuple[int, int]:
        if action == self.stop_action:
            raise ValueError("STOP has no anchor or zoom index")
        zoom_count = len(self.anchor_cfg.zoom_bins)
        return int(action // zoom_count), int(action % zoom_count)

    def roi_for_action(self, action: int) -> np.ndarray:
        anchor_index, zoom_index = self.decode_action(action)
        if anchor_index >= len(self.anchors):
            raise IndexError("Action points to a padded anchor")
        return anchor_to_roi(
            self.anchors[anchor_index],
            self.anchor_cfg.zoom_bins[zoom_index],
            self.image_shape,
            self.anchor_cfg.base_slice_fraction,
            self.anchor_cfg.min_slice_fraction,
            self.anchor_cfg.max_slice_fraction,
            self.anchor_cfg.context_factor,
        )

    def action_mask(self) -> np.ndarray:
        mask = np.zeros((self.action_count,), dtype=bool)
        if not self.done:
            zoom_count = len(self.anchor_cfg.zoom_bins)
            attempted = np.asarray(self.attempted_rois, dtype=np.float32).reshape(-1, 4)
            for anchor_index in range(len(self.anchors)):
                if anchor_index in self.selected_anchors:
                    continue
                for zoom_index in range(zoom_count):
                    action = anchor_index * zoom_count + zoom_index
                    roi = self.roi_for_action(action)
                    if len(attempted):
                        overlap = float(iou_matrix(roi[None, :], attempted).max())
                        if overlap >= self.env_cfg.action_overlap_threshold:
                            continue
                    mask[action] = True
        mask[self.stop_action] = True
        return mask

    def state(self) -> np.ndarray:
        return build_state(
            self.anchors,
            self.full_detections,
            self.attempted_rois,
            self.accepted_rois,
            self.selected_anchors,
            self.image_shape,
            self.last_utility,
            self.anchor_cfg,
            self.env_cfg,
        )

    def overlap_with_history(self, roi: np.ndarray) -> float:
        if not self.attempted_rois:
            return 0.0
        return float(
            iou_matrix(
                np.asarray(roi, dtype=np.float32)[None, :],
                np.asarray(self.attempted_rois, dtype=np.float32),
            ).max()
        )

    def record(self, action: int, roi: np.ndarray, accepted: bool, utility: float) -> None:
        anchor_index, _ = self.decode_action(action)
        self.selected_anchors.add(anchor_index)
        self.attempted_rois.append(np.asarray(roi, dtype=np.float32))
        if accepted:
            self.accepted_rois.append(np.asarray(roi, dtype=np.float32))
        self.last_utility = float(utility)
