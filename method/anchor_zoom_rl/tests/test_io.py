from __future__ import annotations

import numpy as np

from anchor_zoom_rl.core.types import Detections
from anchor_zoom_rl.runtime.io import save_visdrone_predictions


def test_visdrone_writer_sorts_scores_and_skips_invalid_boxes(tmp_path) -> None:
    detections = Detections(
        np.asarray(
            [
                [10.0, 20.0, 30.0, 50.0],
                [1.0, 2.0, 6.0, 9.0],
                [8.0, 8.0, 8.0, 12.0],
            ],
            dtype=np.float32,
        ),
        np.asarray([0.4, 0.9, 0.8], dtype=np.float32),
        np.asarray([0, 2, 1], dtype=np.int64),
    )
    output = tmp_path / "result.txt"

    save_visdrone_predictions(output, detections)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "1.00,2.00,5.00,7.00,0.900000,3,-1,-1",
        "10.00,20.00,20.00,30.00,0.400000,1,-1,-1",
    ]
