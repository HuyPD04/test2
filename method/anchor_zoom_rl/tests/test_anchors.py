from __future__ import annotations

import numpy as np

from anchor_zoom_rl.config import AnchorConfig
from anchor_zoom_rl.core.anchors import generate_anchors
from anchor_zoom_rl.core.types import Detections


def test_generate_anchors_clusters_detections_and_pads_with_grid() -> None:
    detections = Detections(
        boxes=np.asarray(
            [
                [10, 10, 20, 20],
                [22, 10, 32, 20],
                [75, 75, 90, 90],
            ],
            dtype=np.float32,
        ),
        scores=np.asarray([0.1, 0.2, 0.8], dtype=np.float32),
        classes=np.asarray([0, 0, 1], dtype=np.int64),
    )
    cfg = AnchorConfig(top_k=4, cluster_grid_size=2, fallback_grid_size=3)
    anchors = generate_anchors(detections, (100, 100), cfg)

    assert len(anchors) == 4
    assert anchors[0].count == 2
    assert any(anchor.source == "grid" for anchor in anchors)
    assert anchors[0].score >= anchors[1].score
