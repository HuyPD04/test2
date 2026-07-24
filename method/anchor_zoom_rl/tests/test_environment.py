from __future__ import annotations

import numpy as np

from anchor_zoom_rl.config import AnchorConfig, EnvironmentConfig
from anchor_zoom_rl.core.state import state_dimension
from anchor_zoom_rl.core.types import Anchor, Detections
from anchor_zoom_rl.runtime.environment import AnchorZoomEnvironment


def _anchor(cx: float, cy: float) -> Anchor:
    return Anchor(cx, cy, 10, 10, 0.5, 2, 0.2, 0.3, 1.0, 0.001)


def test_action_maps_directly_to_roi_and_masks_selected_anchor() -> None:
    anchor_cfg = AnchorConfig(top_k=2, zoom_bins=(1.0, 2.0), add_fallback_grid=False)
    env = AnchorZoomEnvironment(
        anchors=[_anchor(25, 25), _anchor(75, 75)],
        full_detections=Detections.empty(),
        image_shape=(100, 100),
        anchor_cfg=anchor_cfg,
        env_cfg=EnvironmentConfig(max_crops=2),
    )

    assert env.action_count == 5
    assert env.stop_action == 4
    initial_mask = env.action_mask()
    assert initial_mask.tolist() == [True, True, True, True, True]
    roi = env.roi_for_action(1)
    env.record(1, roi, accepted=True, utility=0.7)
    updated_mask = env.action_mask()

    assert updated_mask[0] == updated_mask[1] == False
    assert updated_mask[4]
    assert env.state().shape == (state_dimension(anchor_cfg),)
