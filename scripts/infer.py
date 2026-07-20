from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.config import load_default_config
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.data import iter_images
from rl_sahi.common.device import print_device_info
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.pipeline import AdaptiveSahiInferencer


def _bool_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive-slice inference and save prediction metadata.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    infer_cfg = cfg.section("infer")
    device = cfg.optional_str("infer", "device")
    policy_device = args.policy_device or cfg.optional_str("infer", "policy_device") or device
    print_device_info("infer", device)
    print_device_info("infer-policy", policy_device)
    target_classes = cfg.target_classes()

    if args.image is not None:
        image_path = args.image if args.image.is_absolute() else ROOT / args.image
        images = [image_path]
        split = args.split
    else:
        if args.split is None:
            raise ValueError("Use --image for one image or --split train/val/test for a dataset split.")
        images = iter_images(cfg.path_value("image_root"), split=args.split, limit=args.limit)
        split = args.split

    if args.checkpoint is None:
        checkpoint = cfg.path_value("checkpoint")
    else:
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    save_visualization = _bool_value(infer_cfg.get("save_visualization", False))
    if args.visualize:
        save_visualization = True
    if args.no_visualize:
        save_visualization = False
    try:
        crop_weights = cfg.path_value("crop_weights")
    except KeyError:
        crop_weights = None
    try:
        full_weights = cfg.path_value("full_weights")
    except KeyError:
        full_weights = None

    inferencer = AdaptiveSahiInferencer(
        weights=cfg.path_value("weights"),
        checkpoint=checkpoint,
        crop_weights=crop_weights,
        full_weights=full_weights,
        cfg=InferenceConfig(
            full_imgsz=int(infer_cfg["full_imgsz"]),
            slice_imgsz=int(infer_cfg["slice_imgsz"]),
            full_conf=float(infer_cfg["full_conf"]),
            output_conf=float(infer_cfg["output_conf"]),
            iou=float(infer_cfg["iou"]),
            merge_iou=float(infer_cfg["merge_iou"]),
            max_det=int(infer_cfg["max_det"]),
            device=device,
            policy_device=policy_device,
            feature_layers=cfg.feature_layers("infer"),
            min_slice_detections=int(infer_cfg.get("min_slice_detections", 1)),
            min_slice_utility=float(infer_cfg.get("min_slice_utility", 0.5)),
            min_new_detection_score=float(infer_cfg.get("min_new_detection_score", 0.45)),
            duplicate_iou=float(infer_cfg.get("duplicate_iou", infer_cfg.get("merge_iou", 0.5))),
            max_slice_attempts=int(infer_cfg.get("max_slice_attempts", 0)),
            roi_prefilter_enabled=_bool_value(infer_cfg.get("roi_prefilter_enabled", False)),
            roi_prefilter_topk=int(infer_cfg.get("roi_prefilter_topk", 3)),
            crop_batch_size=int(infer_cfg.get("crop_batch_size", 1)),
            max_consecutive_rejections=int(infer_cfg.get("max_consecutive_rejections", 0)),
            target_classes=target_classes,
            require_stop_for_acceptance=_bool_value(infer_cfg.get("require_stop_for_acceptance", True)),
            save_predictions=_bool_value(infer_cfg.get("save_predictions", True)) and not args.no_save_predictions,
            save_metadata=_bool_value(infer_cfg.get("save_metadata", True)) and not args.no_metadata,
            save_visualization=save_visualization,
            batched_inference=_bool_value(infer_cfg.get("batched_inference", False)),
            class_mapping=ClassMapping.from_config(cfg.section("classes")),
        ),
    )
    for image_path in images:
        meta = inferencer.infer_image(
            image_path=image_path,
            out_dir=cfg.path_value("infer_out_dir"),
            cache_root=cfg.path_value("cache_root") if split is not None else None,
            split=split,
            use_cache=_bool_value(infer_cfg["use_cache"]) and not args.no_cache,
        )
        timing = meta.get("timing", {})
        total_ms = float(timing.get("total_ms", 0.0))
        crop_ms = float(timing.get("crop_inference_ms", 0.0))
        print(
            f"[infer] {image_path.name}: {meta['detections']} boxes, "
            f"slices={meta['num_slices']} attempts={meta['num_attempts']} "
            f"candidates={meta.get('num_roi_candidates', 0)} "
            f"prefiltered={meta.get('num_roi_prefilter_dropped', 0)} "
            f"crops={meta['num_crop_predictions']}/{meta['num_crop_batches']} "
            f"time={total_ms:.1f}ms crop={crop_ms:.1f}ms"
        )
        print(
            f"[timing] {image_path.name}: "
            f"load={float(timing.get('image_read_ms', 0.0)):.1f}ms "
            f"full={float(timing.get('initial_detection_ms', 0.0)):.1f}ms "
            f"(read={float(timing.get('initial_image_read_ms', 0.0)):.1f} "
            f"call={float(timing.get('initial_yolo_wall_ms', 0.0)):.1f} "
            f"pre={float(timing.get('initial_preprocess_ms', 0.0)):.1f} "
            f"yolo={float(timing.get('initial_yolo_inference_ms', 0.0)):.1f} "
            f"post={float(timing.get('initial_postprocess_ms', 0.0)):.1f} "
            f"feat={float(timing.get('initial_feature_extract_ms', 0.0)):.1f} "
            f"d2h={float(timing.get('initial_result_transfer_ms', 0.0)):.1f}) "
            f"crop={crop_ms:.1f}ms "
            f"(read={float(timing.get('crop_image_read_ms', 0.0)):.1f} "
            f"extract={float(timing.get('crop_extract_ms', 0.0)):.1f} "
            f"call={float(timing.get('crop_yolo_wall_ms', 0.0)):.1f} "
            f"pre={float(timing.get('crop_preprocess_ms', 0.0)):.1f} "
            f"yolo={float(timing.get('crop_yolo_inference_ms', 0.0)):.1f} "
            f"post={float(timing.get('crop_postprocess_ms', 0.0)):.1f} "
            f"d2h={float(timing.get('crop_result_transfer_ms', 0.0)):.1f}) "
            f"rollout={float(timing.get('rollout_ms', 0.0)):.1f}ms "
            f"(static={float(timing.get('rollout_static_ms', 0.0)):.1f} "
            f"env={float(timing.get('rollout_env_init_ms', 0.0)):.1f} "
            f"state={float(timing.get('rollout_state_ms', 0.0)):.1f} "
            f"valid={float(timing.get('rollout_valid_ms', 0.0)):.1f} "
            f"policy={float(timing.get('rollout_policy_ms', 0.0)):.1f} "
            f"step={float(timing.get('rollout_step_ms', 0.0)):.1f}) "
            f"prefilter={float(timing.get('roi_prefilter_ms', 0.0)):.1f}ms "
            f"merge={float(timing.get('merge_ms', 0.0)):.1f}ms "
            f"write={float(timing.get('write_outputs_ms', 0.0)):.1f}ms"
        )


if __name__ == "__main__":
    main()
