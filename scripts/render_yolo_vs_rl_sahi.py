from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import read_image
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.pipeline import AdaptiveSahiInferencer, get_initial_detection
from rl_sahi.inference.visualize import draw_boxes


YOLO_BOX_COLOR = (0, 255, 0)  # Green, BGR.
SLICE_COLOR = (0, 165, 255)  # Orange, BGR.


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _optional_float(value: object, default: float | None = None) -> float | None:
    raw = default if value is None else value
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null", "false", "off"}:
        return None
    return float(raw)


def _load_prediction_boxes(path: Path) -> np.ndarray:
    """Read ``class score x1 y1 x2 y2 source`` predictions, discarding non-box fields."""
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) >= 6:
            boxes.append([float(value) for value in values[2:6]])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def _slice_rois(metadata: dict) -> np.ndarray:
    rois = [
        slice_info["roi"]
        for slice_info in metadata.get("slices", [])
        if bool(slice_info.get("accepted", False)) and "roi" in slice_info
    ]
    return np.asarray(rois, dtype=np.float32).reshape(-1, 4)


def _path_from_arg(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _make_inference_config(config) -> InferenceConfig:
    infer_cfg = config.section("infer")
    return InferenceConfig(
        full_imgsz=int(infer_cfg["full_imgsz"]),
        slice_imgsz=int(infer_cfg["slice_imgsz"]),
        full_conf=float(infer_cfg["full_conf"]),
        output_conf=float(infer_cfg["output_conf"]),
        iou=float(infer_cfg["iou"]),
        merge_iou=float(infer_cfg["merge_iou"]),
        max_det=int(infer_cfg["max_det"]),
        device=config.optional_str("infer", "device"),
        policy_device=config.optional_str("infer", "policy_device"),
        feature_layers=config.feature_layers("infer"),
        spatial_feature_layers=config.spatial_feature_layers("infer"),
        min_slice_detections=int(infer_cfg.get("min_slice_detections", 1)),
        min_slice_utility=float(infer_cfg.get("min_slice_utility", 0.5)),
        min_new_detection_score=float(infer_cfg.get("min_new_detection_score", 0.45)),
        duplicate_iou=float(infer_cfg.get("duplicate_iou", infer_cfg.get("merge_iou", 0.5))),
        boundary_margin=float(infer_cfg.get("boundary_margin", 2.0)),
        append_novel_only=_as_bool(infer_cfg.get("append_novel_only", False)),
        cross_class_duplicate_iou=_optional_float(infer_cfg.get("cross_class_duplicate_iou"), 0.85),
        cross_class_duplicate_ios=_optional_float(infer_cfg.get("cross_class_duplicate_ios"), 0.95),
        max_slice_attempts=int(infer_cfg.get("max_slice_attempts", 0)),
        roi_prefilter_enabled=_as_bool(infer_cfg.get("roi_prefilter_enabled", False)),
        roi_prefilter_topk=int(infer_cfg.get("roi_prefilter_topk", 3)),
        crop_batch_size=int(infer_cfg.get("crop_batch_size", 1)),
        max_consecutive_rejections=int(infer_cfg.get("max_consecutive_rejections", 0)),
        target_classes=config.target_classes(),
        require_stop_for_acceptance=_as_bool(infer_cfg.get("require_stop_for_acceptance", True)),
        # These artifacts are required to render the RL-SAHI image below.
        save_predictions=True,
        save_metadata=True,
        save_visualization=False,
        batched_inference=_as_bool(infer_cfg.get("batched_inference", False)),
        use_wbf=_as_bool(infer_cfg.get("use_wbf", False)),
        nms_type=str(infer_cfg.get("nms_type", "standard")),
        gate_nms_type=str(infer_cfg.get("gate_nms_type", "standard")),
        class_mapping=ClassMapping.from_config(config.section("classes")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render label-free YOLO and RL-SAHI bounding-box images for one input image."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/render_comparison"))
    parser.add_argument("--config", type=Path, default=None, help="Project YAML config.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="RL policy checkpoint.")
    parser.add_argument("--yolo-imgsz", type=int, default=1280)
    parser.add_argument("--yolo-conf", type=float, default=0.5)
    parser.add_argument("--yolo-iou", type=float, default=0.5)
    args = parser.parse_args()

    if args.yolo_imgsz <= 0:
        parser.error("--yolo-imgsz must be positive")
    if not 0.0 <= args.yolo_conf <= 1.0:
        parser.error("--yolo-conf must be between 0 and 1")
    if not 0.0 <= args.yolo_iou <= 1.0:
        parser.error("--yolo-iou must be between 0 and 1")

    image_path = _path_from_arg(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    out_dir = _path_from_arg(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_default_config(args.config, ROOT)
    checkpoint = config.path_value("checkpoint") if args.checkpoint is None else _path_from_arg(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RL checkpoint not found: {checkpoint}")

    try:
        crop_weights = config.path_value("crop_weights")
    except KeyError:
        crop_weights = None
    try:
        full_weights = config.path_value("full_weights")
    except KeyError:
        full_weights = None

    inferencer = AdaptiveSahiInferencer(
        weights=config.path_value("weights"),
        checkpoint=checkpoint,
        crop_weights=crop_weights,
        full_weights=full_weights,
        cfg=_make_inference_config(config),
    )
    image = read_image(image_path)

    # Image 1: exact YOLO baseline requested by the user.
    yolo_det = get_initial_detection(
        model=inferencer.full_yolo,
        weights=inferencer.full_weights or inferencer.weights,
        image_path=image_path,
        weights_imgsz=args.yolo_imgsz,
        full_conf=args.yolo_conf,
        full_iou=args.yolo_iou,
        max_det=inferencer.cfg.max_det,
        device=inferencer.cfg.device,
        feature_layers=inferencer.cfg.feature_layers,
        spatial_feature_layers=inferencer.cfg.spatial_feature_layers,
        aux_grid_size=inferencer.state_cfg.grid_size,
        spatial_feature_channels=inferencer.state_cfg.spatial_feature_channels,
        use_cache=False,
        source_image=image,
    )
    yolo_image = image.copy()
    draw_boxes(yolo_image, yolo_det.boxes, color=YOLO_BOX_COLOR, classes=None, thickness=2)
    yolo_path = out_dir / (
        f"yolo_{args.yolo_imgsz}_conf{args.yolo_conf:.2f}_iou{args.yolo_iou:.2f}_{image_path.stem}.jpg"
    )
    if not cv2.imwrite(str(yolo_path), yolo_image):
        raise RuntimeError(f"Could not write image: {yolo_path}")

    # Image 2: final RL-SAHI boxes plus accepted ROIs only, if any.
    # The inference pipeline needs predictions/metadata on disk, but the user-facing
    # output remains exactly the two rendered images.
    with TemporaryDirectory(prefix="rl_sahi_render_", dir=out_dir) as temporary_dir:
        metadata = inferencer.infer_image(image_path, out_dir=Path(temporary_dir), use_cache=False)
        final_boxes = _load_prediction_boxes(Path(metadata["prediction_file"]))
        rois = _slice_rois(metadata)

    rl_image = image.copy()
    draw_boxes(rl_image, rois, color=SLICE_COLOR, classes=None, thickness=2)
    draw_boxes(rl_image, final_boxes, color=YOLO_BOX_COLOR, classes=None, thickness=2)
    rl_path = out_dir / f"rl_sahi_{image_path.stem}.jpg"
    if not cv2.imwrite(str(rl_path), rl_image):
        raise RuntimeError(f"Could not write image: {rl_path}")

    print(f"[render] YOLO: {yolo_path} ({len(yolo_det.boxes)} boxes)")
    print(f"[render] RL-SAHI: {rl_path} ({len(final_boxes)} boxes, {len(rois)} slice ROIs)")


if __name__ == "__main__":
    main()
