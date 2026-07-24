from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from anchor_zoom_rl.config import (
    AnchorConfig,
    DetectorConfig,
    EnvironmentConfig,
    InferenceConfig,
    MethodConfig,
    PathsConfig,
    RewardConfig,
    TrainConfig,
)
from anchor_zoom_rl.core.state import state_dimension
from anchor_zoom_rl.core.types import Detections
from anchor_zoom_rl.rl.agent import DQNAgent
from anchor_zoom_rl.runtime.inferencer import AnchorZoomInferencer


class FakeRunner:
    def full(self, image, image_path, split, use_cache):
        return (
            Detections(
                np.asarray([[40, 40, 50, 50]], dtype=np.float32),
                np.asarray([0.1], dtype=np.float32),
                np.asarray([0], dtype=np.int64),
            ),
            2.0,
            False,
        )

    def crop(self, image, image_path, split, roi, use_cache):
        return (
            Detections(
                np.asarray([[42, 42, 54, 54]], dtype=np.float32),
                np.asarray([0.9], dtype=np.float32),
                np.asarray([0], dtype=np.int64),
            ),
            3.0,
            False,
        )


def test_inference_runs_one_direct_anchor_action(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
    anchor_cfg = AnchorConfig(
        top_k=1,
        zoom_bins=(1.0,),
        add_fallback_grid=False,
        cluster_grid_size=2,
    )
    train_cfg = TrainConfig(hidden_dim=32, min_replay=2, batch_size=2)
    checkpoint = tmp_path / "best.pt"
    cfg = MethodConfig(
        paths=PathsConfig(
            weights=tmp_path / "model.pt",
            crop_weights=tmp_path / "model.pt",
            image_root=tmp_path,
            label_root=tmp_path,
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            checkpoint=checkpoint,
        ),
        detector=DetectorConfig(device="cpu", output_confidence=0.25),
        anchors=anchor_cfg,
        environment=EnvironmentConfig(max_crops=1),
        reward=RewardConfig(min_utility=0.1),
        train=train_cfg,
        inference=InferenceConfig(
            save_predictions=False,
            save_metadata=False,
            save_visualization=False,
        ),
    )
    agent = DQNAgent(state_dimension(anchor_cfg), cfg.action_count, train_cfg, "cpu")
    final_layer = agent.online.advantage[-1]
    final_layer.weight.data.zero_()
    final_layer.bias.data[0] = 10.0
    final_layer.bias.data[1] = -10.0
    agent.save(checkpoint, episode=0, environment_steps=0)

    inferencer = AnchorZoomInferencer(cfg, runner=FakeRunner())
    result = inferencer.infer_image(image_path, split="test", save=False)

    assert len(result.attempted_rois) == 1
    assert len(result.accepted_rois) == 1
    assert len(result.detections) == 1
    assert result.actions[0]["type"] == "crop"
    assert abs(
        result.timing["initial_state_ms"]
        + result.timing["method_latency_ms"]
        - result.timing["end_to_end_ms"]
    ) < 1e-6
