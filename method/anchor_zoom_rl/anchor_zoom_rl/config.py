from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class PathsConfig:
    weights: Path
    crop_weights: Path
    image_root: Path
    label_root: Path
    cache_dir: Path
    output_dir: Path
    checkpoint: Path


@dataclass(slots=True)
class DetectorConfig:
    full_imgsz: int = 960
    crop_imgsz: int = 512
    full_confidence: float = 0.01
    crop_confidence: float = 0.05
    output_confidence: float = 0.25
    yolo_iou: float = 0.7
    merge_iou: float = 0.6
    duplicate_iou: float = 0.5
    cross_class_iou: float = 0.85
    cross_class_ios: float | None = None
    cross_class_score_ratio: float = 0.90
    cross_class_groups: tuple[tuple[int, ...], ...] = (
        (0, 1),
        (3, 4, 5, 8),
        (2, 6, 7, 9),
    )
    max_detections: int = 3000
    target_classes: tuple[int, ...] = tuple(range(10))
    device: str = "cuda"


@dataclass(slots=True)
class AnchorConfig:
    top_k: int = 12
    zoom_bins: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0)
    cluster_grid_size: int = 6
    min_confidence: float = 0.01
    low_confidence: float = 0.35
    small_area_ratio: float = 0.0025
    density_norm: float = 8.0
    full_count_norm: float = 100.0
    uncertainty_weight: float = 0.45
    density_weight: float = 0.35
    small_object_weight: float = 0.20
    base_slice_fraction: float = 0.35
    min_slice_fraction: float = 0.12
    max_slice_fraction: float = 0.42
    context_factor: float = 1.6
    add_fallback_grid: bool = True
    fallback_grid_size: int = 3
    fallback_score: float = 0.03


@dataclass(slots=True)
class EnvironmentConfig:
    max_crops: int = 4
    action_overlap_threshold: float = 0.75


@dataclass(slots=True)
class RewardConfig:
    match_iou: float = 0.5
    hard_low_confidence: float = 0.10
    small_area_ratio: float = 0.0025
    min_crop_detections: int = 1
    min_utility: float = 0.15
    min_reliability: float = 0.30
    utility_weight: float = 0.5
    tp_weight: float = 2.0
    hard_tp_weight: float = 0.75
    small_tp_weight: float = 0.5
    fp_weight: float = 1.25
    crop_cost: float = 0.50
    overlap_penalty: float = 0.5
    empty_penalty: float = 0.8
    rejected_penalty: float = 0.5
    stop_bonus: float = 0.25
    stop_early_penalty: float = 0.50
    stop_hard_early_penalty: float = 0.75


@dataclass(slots=True)
class TrainConfig:
    episodes: int = 15000
    batch_size: int = 256
    replay_size: int = 50000
    min_replay: int = 1000
    gamma: float = 0.95
    n_step: int = 3
    learning_rate: float = 0.0001
    hidden_dim: int = 512
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30000
    target_update_interval: int = 250
    soft_update_tau: float = 0.005
    gradient_clip: float = 10.0
    checkpoint_interval: int = 250
    log_interval: int = 25
    eval_interval: int = 1000
    eval_images: int = 256
    eval_ap_weight: float = 1.0
    eval_fp_per_image_weight: float = 0.002
    eval_crop_weight: float = 0.02
    reward_clip: float = 10.0
    sampling_mode: str = "shuffled_epochs"
    seed: int = 42
    resume: bool = False
    cache_full_detections: bool = True
    cache_hard_regions: bool = True
    cache_crop_detections: bool = True


@dataclass(slots=True)
class InferenceConfig:
    save_predictions: bool = True
    save_metadata: bool = True
    save_visualization: bool = False
    cache_full_detections: bool = False
    cache_crop_detections: bool = False


@dataclass(slots=True)
class MethodConfig:
    paths: PathsConfig
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    anchors: AnchorConfig = field(default_factory=AnchorConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @property
    def action_count(self) -> int:
        return self.anchors.top_k * len(self.anchors.zoom_bins) + 1

    @property
    def stop_action(self) -> int:
        return self.action_count - 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_path(config_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section '{key}' must be a mapping")
    return value


def load_config(path: str | Path) -> MethodConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    path_raw = _section(raw, "paths")
    required = (
        "weights",
        "image_root",
        "label_root",
        "cache_dir",
        "output_dir",
        "checkpoint",
    )
    missing = [key for key in required if key not in path_raw]
    if missing:
        raise KeyError(f"Missing path settings: {', '.join(missing)}")
    weights = _resolve_path(config_path.parent, path_raw["weights"])
    paths = PathsConfig(
        weights=weights,
        crop_weights=_resolve_path(config_path.parent, path_raw.get("crop_weights", weights)),
        image_root=_resolve_path(config_path.parent, path_raw["image_root"]),
        label_root=_resolve_path(config_path.parent, path_raw["label_root"]),
        cache_dir=_resolve_path(config_path.parent, path_raw["cache_dir"]),
        output_dir=_resolve_path(config_path.parent, path_raw["output_dir"]),
        checkpoint=_resolve_path(config_path.parent, path_raw["checkpoint"]),
    )

    detector_raw = _section(raw, "detector")
    if "target_classes" in detector_raw:
        detector_raw = {**detector_raw, "target_classes": tuple(detector_raw["target_classes"])}
    if "cross_class_groups" in detector_raw:
        detector_raw = {
            **detector_raw,
            "cross_class_groups": tuple(
                tuple(int(value) for value in group)
                for group in detector_raw["cross_class_groups"]
            ),
        }
    anchor_raw = _section(raw, "anchors")
    if "zoom_bins" in anchor_raw:
        anchor_raw = {**anchor_raw, "zoom_bins": tuple(anchor_raw["zoom_bins"])}
    cfg = MethodConfig(
        paths=paths,
        detector=DetectorConfig(**detector_raw),
        anchors=AnchorConfig(**anchor_raw),
        environment=EnvironmentConfig(**_section(raw, "environment")),
        reward=RewardConfig(**_section(raw, "reward")),
        train=TrainConfig(**_section(raw, "train")),
        inference=InferenceConfig(**_section(raw, "inference")),
    )
    if cfg.anchors.top_k <= 0 or not cfg.anchors.zoom_bins:
        raise ValueError("anchors.top_k and anchors.zoom_bins must define a non-empty action space")
    if cfg.environment.max_crops <= 0:
        raise ValueError("environment.max_crops must be positive")
    if cfg.train.sampling_mode not in {
        "shuffled_epochs",
        "random_with_replacement",
    }:
        raise ValueError(
            "train.sampling_mode must be 'shuffled_epochs' or "
            "'random_with_replacement'"
        )
    return cfg
