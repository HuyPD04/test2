from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

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
from anchor_zoom_rl.core.hard_regions import build_hard_regions
from anchor_zoom_rl.core.types import Detections
from anchor_zoom_rl.runtime.trainer import (
    TRAINING_SCHEMA_VERSION,
    AnchorZoomTrainer,
)


class FakeTrainingRunner:
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
                np.asarray([[45, 45, 55, 55]], dtype=np.float32),
                np.asarray([0.9], dtype=np.float32),
                np.asarray([0], dtype=np.int64),
            ),
            3.0,
            False,
        )

    def hard_regions(
        self,
        full_detections,
        ground_truth,
        image_path,
        label_path,
        split,
        use_cache,
    ):
        return (
            build_hard_regions(
                full_detections,
                ground_truth,
                match_iou=0.5,
                low_confidence=0.25,
            ),
            False,
        )


def test_training_writes_schema3_metrics_and_best_checkpoints(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    for split in ("train", "val"):
        (image_root / split).mkdir(parents=True)
        (label_root / split).mkdir(parents=True)
        cv2.imwrite(
            str(image_root / split / "sequence_0001.jpg"),
            np.zeros((100, 100, 3), dtype=np.uint8),
        )
        (label_root / split / "sequence_0001.txt").write_text(
            "0 0.5 0.5 0.1 0.1\n",
            encoding="utf-8",
        )

    output_dir = tmp_path / "run"
    anchor_cfg = AnchorConfig(
        top_k=1,
        zoom_bins=(1.0,),
        add_fallback_grid=False,
        cluster_grid_size=2,
    )
    train_cfg = TrainConfig(
        episodes=1,
        hidden_dim=32,
        batch_size=1,
        min_replay=1,
        replay_size=4,
        n_step=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        eval_interval=1,
        eval_images=1,
        checkpoint_interval=1,
    )
    cfg = MethodConfig(
        paths=PathsConfig(
            weights=tmp_path / "model.pt",
            crop_weights=tmp_path / "model.pt",
            image_root=image_root,
            label_root=label_root,
            cache_dir=tmp_path / "cache",
            output_dir=output_dir,
            checkpoint=output_dir / "checkpoints" / "best.pt",
        ),
        detector=DetectorConfig(device="cpu", output_confidence=0.25),
        anchors=anchor_cfg,
        environment=EnvironmentConfig(max_crops=1),
        reward=RewardConfig(
            hard_low_confidence=0.25,
            min_utility=0.1,
        ),
        train=train_cfg,
        inference=InferenceConfig(),
    )
    trainer = AnchorZoomTrainer(cfg, runner=FakeTrainingRunner())
    final_layer = trainer.agent.online.advantage[-1]
    final_layer.weight.data.zero_()
    final_layer.bias.data[0] = 10.0
    final_layer.bias.data[1] = -10.0
    trainer.agent.target.load_state_dict(trainer.agent.online.state_dict())

    trainer.train(1)

    checkpoint_dir = output_dir / "checkpoints"
    for name in (
        "best.pt",
        "best_tradeoff.pt",
        "best_ap.pt",
        "best_hard_recall.pt",
        "latest.pt",
    ):
        assert (checkpoint_dir / name).exists()
    payload = torch.load(
        checkpoint_dir / "latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert payload["training_schema_version"] == TRAINING_SCHEMA_VERSION
    assert payload["best_metrics"]["ap50"] >= 0.0
    eval_rows = (output_dir / "eval.csv").read_text(encoding="utf-8").splitlines()
    assert "hard_recall" in eval_rows[0]
    assert "attempted_hard_coverage" in eval_rows[0]
    assert len(eval_rows) == 2

    cfg.train.resume = True
    resumed = AnchorZoomTrainer(cfg, runner=FakeTrainingRunner())
    assert resumed.start_episode == 2
    assert len(resumed.replay) == 1
    assert resumed.best_ap == 1.0
