from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.cache import detection_cache_metadata, hard_region_cache_metadata
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.device import print_device_info
from rl_sahi.common.data import iter_images
from rl_sahi.eval.benchmark import BenchmarkConfig
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.rl.env_config import EnvConfig
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.rl.trainer import TrainConfig
from rl_sahi.rl.batched_trainer import batched_train_dqn


def _bool_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN to choose one adaptive slice from cached YOLO state.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Checkpoint/log directory; use a distinct directory for each ablation variant.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--spatial-feature",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable YOLO spatial feature maps in the DQN state (default: config, enabled).",
    )
    parser.add_argument(
        "--detector-cues",
        "--detection-map",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable detection maps, objectness map, and detection summaries in the DQN state (default: config, enabled).",
    )
    parser.add_argument(
        "--history",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable visited/attempted/accepted ROI memory in the DQN state (default: config, enabled).",
    )
    parser.add_argument(
        "--outcome-reward",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable terminal reward from actual crop detector outcomes (default: config, enabled).",
    )
    parser.add_argument(
        "--cost-overlap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable compute, scale, and overlap reward penalties (default: config, enabled).",
    )
    parser.add_argument(
        "--action-mask",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Mask stalled/redundant actions in exploration, policy, and Bellman targets (default: config, enabled).",
    )
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    train_cfg = cfg.dataclass_instance("train", TrainConfig)
    env_cfg = cfg.dataclass_instance("env", EnvConfig)
    state_cfg = cfg.dataclass_instance("state", StateConfig)
    if args.spatial_feature is not None:
        state_cfg.use_spatial_features = bool(args.spatial_feature)
    if args.detector_cues is not None:
        state_cfg.use_detector_cues = bool(args.detector_cues)
    if args.history is not None:
        state_cfg.use_history = bool(args.history)
    if args.outcome_reward is not None:
        train_cfg.use_crop_outcome_reward = bool(args.outcome_reward)
    if args.cost_overlap is not None:
        env_cfg.use_cost_overlap_reward = bool(args.cost_overlap)
    if args.action_mask is not None:
        env_cfg.use_action_mask = bool(args.action_mask)
    detect_cfg = cfg.section("detect")
    hard_cfg = cfg.section("hard_region")
    infer_cfg = cfg.section("infer")
    benchmark_cfg = cfg.section("benchmark")
    target_classes = cfg.target_classes()
    class_mapping = ClassMapping.from_config(cfg.section("classes"))
    if not train_cfg.allow_sequence_overlap and args.split == "train":
        train_sequences = {
            image.stem.split("_", 1)[0]
            for image in iter_images(cfg.path_value("image_root"), split="train")
        }
        val_sequences = {
            image.stem.split("_", 1)[0]
            for image in iter_images(cfg.path_value("image_root"), split=train_cfg.val_split)
        }
        overlap = sorted(train_sequences & val_sequences)
        if overlap:
            preview = ", ".join(overlap[:10])
            raise ValueError(
                f"Train/validation sequence leakage detected ({len(overlap)} shared ids: {preview}). "
                "Regroup the dataset by filename prefix before retraining, or set "
                "train.allow_sequence_overlap=true only if the prefix is not a sequence id."
            )
    if args.episodes is not None:
        train_cfg.episodes = args.episodes
    if args.resume is not None:
        train_cfg.resume = bool(args.resume)
    device_name = args.device or cfg.optional_str("train", "device")
    print_device_info("train", device_name)
    print(
        "[train] components: "
        f"spatial_feature={state_cfg.use_spatial_features} "
        f"detector_cues={state_cfg.use_detector_cues} "
        f"history={state_cfg.use_history} "
        f"outcome_reward={train_cfg.use_crop_outcome_reward} "
        f"cost_overlap={env_cfg.use_cost_overlap_reward} "
        f"action_mask={env_cfg.use_action_mask}"
    )
    detection_metadata = detection_cache_metadata(
        weights=cfg.path_value("weights"),
        imgsz=int(detect_cfg["imgsz"]),
        conf=float(detect_cfg["conf"]),
        iou=float(detect_cfg["iou"]),
        max_det=int(detect_cfg["max_det"]),
        feature_layers=cfg.feature_layers("detect"),
        aux_grid_size=int(state_cfg.grid_size),
        spatial_feature_channels=int(state_cfg.spatial_feature_channels),
    )
    hard_metadata = hard_region_cache_metadata(
        detection_metadata=detection_metadata,
        small_area_ratio=float(hard_cfg["small_area_ratio"]),
        small_area_percentile=(
            None if hard_cfg.get("small_area_percentile") in (None, "")
            else float(hard_cfg["small_area_percentile"])
        ),
        match_iou=float(hard_cfg["match_iou"]),
        min_detect_score=float(hard_cfg["min_detect_score"]),
        target_classes=target_classes,
        class_mapping=class_mapping,
    )

    out_dir = cfg.path_value("dqn_out_dir") if args.out_dir is None else args.out_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    try:
        crop_weights = cfg.path_value("crop_weights")
    except KeyError:
        crop_weights = None
    try:
        full_weights = cfg.path_value("full_weights")
    except KeyError:
        full_weights = None
    try:
        annotation_root = cfg.path_value("annotation_root")
    except KeyError:
        annotation_root = None
    checkpoint = batched_train_dqn(
        image_root=cfg.path_value("image_root"),
        cache_root=cfg.path_value("cache_root"),
        split=args.split,
        out_dir=out_dir,
        cfg=train_cfg,
        env_cfg=env_cfg,
        state_cfg=state_cfg,
        limit=args.limit,
        device_name=device_name,
        detection_metadata=detection_metadata,
        hard_region_metadata=hard_metadata,
        target_classes=target_classes,
        class_mapping=class_mapping,
        label_root=cfg.path_value("label_root"),
        annotation_root=annotation_root,
        eval_weights=cfg.path_value("weights"),
        eval_full_weights=full_weights,
        eval_crop_weights=crop_weights,
        infer_cfg=InferenceConfig(
            full_imgsz=int(infer_cfg["full_imgsz"]),
            slice_imgsz=int(infer_cfg["slice_imgsz"]),
            full_conf=float(infer_cfg["full_conf"]),
            output_conf=float(infer_cfg["output_conf"]),
            iou=float(infer_cfg["iou"]),
            merge_iou=float(infer_cfg["merge_iou"]),
            max_det=int(infer_cfg["max_det"]),
            device=device_name or cfg.optional_str("infer", "device"),
            policy_device=cfg.optional_str("infer", "policy_device"),
            feature_layers=cfg.feature_layers("infer"),
            min_slice_detections=int(infer_cfg.get("min_slice_detections", 1)),
            min_slice_utility=float(infer_cfg.get("min_slice_utility", 0.5)),
            min_new_detection_score=float(infer_cfg.get("min_new_detection_score", 0.45)),
            duplicate_iou=float(infer_cfg.get("duplicate_iou", infer_cfg.get("merge_iou", 0.5))),
            max_slice_attempts=int(infer_cfg.get("max_slice_attempts", 0)),
            roi_prefilter_enabled=bool(infer_cfg.get("roi_prefilter_enabled", False)),
            roi_prefilter_topk=int(infer_cfg.get("roi_prefilter_topk", 3)),
            crop_batch_size=int(infer_cfg.get("crop_batch_size", 1)),
            max_consecutive_rejections=int(infer_cfg.get("max_consecutive_rejections", 0)),
            target_classes=target_classes,
            require_stop_for_acceptance=bool(infer_cfg.get("require_stop_for_acceptance", True)),
            save_predictions=False,
            save_metadata=False,
            save_visualization=False,
            batched_inference=_bool_value(infer_cfg.get("batched_inference", False)),
            use_wbf=_bool_value(infer_cfg.get("use_wbf", False)),
            nms_type=str(infer_cfg.get("nms_type", "standard")),
            class_mapping=class_mapping,
        ),
        bench_cfg=BenchmarkConfig(
            output_conf=float(benchmark_cfg.get("output_conf", 0.01)),
            iou_threshold=float(benchmark_cfg.get("iou_threshold", 0.5)),
            fixed_slice_fraction=float(benchmark_cfg.get("fixed_slice_fraction", 0.35)),
            fixed_overlap=float(benchmark_cfg.get("fixed_overlap", 0.2)),
            small_area_ratio=(
                None if benchmark_cfg.get("small_area_ratio") in (None, "")
                else float(benchmark_cfg["small_area_ratio"])
            ),
            small_area_percentile=(
                None if benchmark_cfg.get("small_area_percentile") in (None, "")
                else float(benchmark_cfg["small_area_percentile"])
            ),
            sampling=str(benchmark_cfg.get("sampling", "stratified")),
            seed=int(benchmark_cfg.get("seed", 42)),
            eval_max_detections=int(benchmark_cfg.get("eval_max_detections", 500)),
            ignore_overlap_threshold=float(benchmark_cfg.get("ignore_overlap_threshold", 0.5)),
            target_classes=target_classes,
            class_mapping=class_mapping,
        ),
        eval_use_cache=bool(infer_cfg.get("use_cache", True)),
    )
    print(f"[train] best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
