from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# Set project root and add src to sys.path (strictly avoiding D:\RL-SAHI\method)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import read_image, read_yolo_labels, image_to_label_path
from rl_sahi.common.boxes import as_boxes
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.hard_region.regions import build_hard_region_cache
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.pipeline import AdaptiveSahiInferencer, get_initial_detection
from rl_sahi.inference.visualize import draw_boxes, draw_detections


def download_image_from_url(url: str, dest_path: Path) -> Path:
    """Download an image from an HTTP/HTTPS URL with proper User-Agent headers."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching image from URL: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
            out_file.write(response.read())
    except Exception as e:
        try:
            import requests
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            dest_path.write_bytes(resp.content)
        except Exception as err:
            raise RuntimeError(f"Failed to download image from URL {url}: {err}") from err
    return dest_path


def resolve_input_image(link: str, root: Path, tmp_dir: Path) -> Path:
    """Resolve input link whether it is an HTTP/HTTPS URL or a local file path."""
    link_str = str(link).strip()
    if link_str.startswith("http://") or link_str.startswith("https://"):
        filename = link_str.split("/")[-1].split("?")[0]
        if not filename or "." not in filename:
            filename = f"downloaded_{int(time.time())}.jpg"
        dest_path = tmp_dir / filename
        return download_image_from_url(link_str, dest_path)
    
    path = Path(link_str)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"Input image not found at: {path}")
    return path


def get_robust_label_path(image_path: Path, image_root: Path, label_root: Path) -> Path:
    """Safely find corresponding ground truth label file, even if image was downloaded to tmp."""
    try:
        return image_to_label_path(image_path, image_root, label_root)
    except ValueError:
        # If image is outside image_root (e.g. downloaded URL), search label_root by stem
        stem = image_path.stem
        matches = list(label_root.rglob(f"{stem}.txt"))
        if matches:
            return matches[0]
        return label_root / f"{stem}.txt"


def _load_prediction_txt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes: list[list[float]] = []
    scores: list[float] = []
    classes: list[float] = []
    sources: list[int] = []
    if not path.exists():
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        classes.append(float(parts[0]))
        scores.append(float(parts[1]))
        boxes.append([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
        sources.append(int(parts[6]))
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(scores, dtype=np.float32),
        np.asarray(classes, dtype=np.float32),
        np.asarray(sources, dtype=np.int32),
    )


def add_header(
    image: np.ndarray,
    title: str,
    header_height: int = 45,
    bg_color: tuple[int, int, int] = (30, 30, 30),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Add a sleek header banner above an image."""
    h, w = image.shape[:2]
    header = np.full((header_height, w, 3), bg_color, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    size, _ = cv2.getTextSize(title, font, scale, thickness)
    text_x = max(15, (w - size[0]) // 2)
    text_y = (header_height + size[1]) // 2
    cv2.putText(header, title, (text_x, text_y), font, scale, text_color, thickness, cv2.LINE_AA)
    return np.vstack([header, image])


def create_info_panel(width: int, height: int, stats: list[tuple[str, str]]) -> np.ndarray:
    """Create a statistics summary panel for grid completion."""
    panel = np.full((height, width, 3), (25, 25, 25), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 50
    cv2.putText(panel, "RL-SAHI INFERENCE STATS", (25, y), font, 0.75, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.line(panel, (25, y + 15), (width - 25, y + 15), (100, 100, 100), 1)
    y += 50
    for label, val in stats:
        text = f"{label}: {val}"
        cv2.putText(panel, text, (25, y), font, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        y += 35
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 5 RL-SAHI visual outputs from an image link/path.")
    parser.add_argument("link", nargs="?", default="data/raw/images/test/0000006_00611_d_0000002.jpg",
                        help="Path or HTTP/HTTPS URL of the input image")
    parser.add_argument("--out-dir", type=Path, default=Path("artifact/output"),
                        help="Directory to save the generated output images")
    parser.add_argument("--config", type=Path, default=None, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to RL checkpoint")
    parser.add_argument("--conf", type=float, default=0.6, help="Confidence threshold for YOLO detection (Image 1)")
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = ROOT / "tmp" / "downloads"

    # 1. Resolve image input link
    image_path = resolve_input_image(args.link, ROOT, tmp_dir)
    print(f"[main] Processing image: {image_path}")
    img = read_image(image_path)
    img_h, img_w = img.shape[:2]

    # 2. Initialize Inferencer
    checkpoint = args.checkpoint or cfg.path_value("checkpoint")
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    try:
        crop_weights = cfg.path_value("crop_weights")
    except KeyError:
        crop_weights = None
    try:
        full_weights = cfg.path_value("full_weights")
    except KeyError:
        full_weights = None

    print(f"[main] Loading AdaptiveSahiInferencer with checkpoint: {checkpoint}")
    infer_cfg = cfg.section("infer")
    class_mapping = ClassMapping.from_config(cfg.section("classes"))
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
            device=infer_cfg.get("device"),
            policy_device=infer_cfg.get("policy_device", infer_cfg.get("device")),
            feature_layers=cfg.feature_layers("infer"),
            spatial_feature_layers=cfg.spatial_feature_layers("infer"),
            min_slice_detections=int(infer_cfg.get("min_slice_detections", 1)),
            min_slice_utility=float(infer_cfg.get("min_slice_utility", 0.5)),
            min_new_detection_score=float(infer_cfg.get("min_new_detection_score", 0.45)),
            duplicate_iou=float(infer_cfg.get("duplicate_iou", infer_cfg.get("merge_iou", 0.5))),
            cross_class_duplicate_iou=float(infer_cfg.get("cross_class_duplicate_iou", 0.85)) if infer_cfg.get("cross_class_duplicate_iou") is not None else None,
            cross_class_duplicate_ios=float(infer_cfg.get("cross_class_duplicate_ios", 0.95)) if infer_cfg.get("cross_class_duplicate_ios") is not None else None,
            max_slice_attempts=int(infer_cfg.get("max_slice_attempts", 0)),
            roi_prefilter_enabled=bool(infer_cfg.get("roi_prefilter_enabled", False)),
            roi_prefilter_topk=int(infer_cfg.get("roi_prefilter_topk", 3)),
            crop_batch_size=int(infer_cfg.get("crop_batch_size", 1)),
            max_consecutive_rejections=int(infer_cfg.get("max_consecutive_rejections", 0)),
            target_classes=cfg.target_classes(),
            require_stop_for_acceptance=bool(infer_cfg.get("require_stop_for_acceptance", True)),
            save_predictions=True,
            save_metadata=True,
            save_visualization=False,
            batched_inference=bool(infer_cfg.get("batched_inference", False)),
            use_wbf=bool(infer_cfg.get("use_wbf", False)),
            nms_type=str(infer_cfg.get("nms_type", "standard")),
            class_mapping=class_mapping,
        ),
    )

    # 3. Load Ground Truth Labels
    label_path = get_robust_label_path(image_path, cfg.path_value("image_root"), cfg.path_value("label_root"))
    gt_classes, gt_boxes = read_yolo_labels(label_path, (img_h, img_w))
    gt_classes = class_mapping.map_label_classes(gt_classes)
    if cfg.target_classes():
        mask = np.isin(gt_classes.astype(np.int64), np.asarray(cfg.target_classes(), dtype=np.int64))
        gt_classes, gt_boxes = gt_classes[mask], gt_boxes[mask]

    # -------------------------------------------------------------------------
    # IMAGE 1: YOLO Detect (conf >= args.conf), green bbox, NO conf and NO class
    # -------------------------------------------------------------------------
    print(f"[main] Generating Image 1: YOLO full detection (conf {args.conf})...")
    initial_timing: dict[str, float] = {}
    det = get_initial_detection(
        model=inferencer.full_yolo,
        weights=inferencer.full_weights or inferencer.weights,
        image_path=image_path,
        weights_imgsz=cfg.section("infer")["full_imgsz"],
        full_conf=0.01,
        full_iou=cfg.section("infer")["iou"],
        max_det=cfg.section("infer")["max_det"],
        device=cfg.section("infer")["device"],
        feature_layers=cfg.feature_layers("infer"),
        spatial_feature_layers=cfg.spatial_feature_layers("infer"),
        aux_grid_size=inferencer.state_cfg.grid_size,
        spatial_feature_channels=inferencer.state_cfg.spatial_feature_channels,
        use_cache=False,
        timing=initial_timing,
        source_image=img,
    )
    conf_mask = det.scores >= args.conf
    boxes_thresh = det.boxes[conf_mask]

    img1 = img.copy()
    draw_boxes(img1, boxes_thresh, color=(0, 255, 0), classes=None, thickness=2)
    conf_str = str(args.conf).replace(".", "")
    path1 = out_dir / f"1_yolo_detect_conf{conf_str}_{image_path.stem}.jpg"
    cv2.imwrite(str(path1), img1)
    print(f"  -> Saved: {path1} ({len(boxes_thresh)} boxes)")

    # -------------------------------------------------------------------------
    # IMAGE 2: Hard Region from Image 1, in red (excluding Image 1 detections)
    # -------------------------------------------------------------------------
    print("[main] Generating Image 2: Hard regions...")
    hard_cfg = cfg.section("hard_region")
    hard_cache = build_hard_region_cache(
        image_path=image_path,
        image_root=image_path.parent if not image_path.is_relative_to(cfg.path_value("image_root")) else cfg.path_value("image_root"),
        label_root=label_path.parent if not image_path.is_relative_to(cfg.path_value("image_root")) else cfg.path_value("label_root"),
        detection_boxes=det.boxes,
        detection_scores=det.scores,
        image_shape=(img_h, img_w),
        detection_classes=det.classes,
        small_area_ratio=float(hard_cfg.get("small_area_ratio", 0.01)),
        match_iou=float(hard_cfg.get("match_iou", 0.4)),
        min_detect_score=float(hard_cfg.get("min_detect_score", 0.25)),
        target_classes=cfg.target_classes(),
        class_mapping=class_mapping,
    )
    img2 = img.copy()
    draw_boxes(img2, hard_cache.hard_boxes, color=(0, 0, 255), classes=None, thickness=2)
    path2 = out_dir / f"2_hard_regions_red_{image_path.stem}.jpg"
    cv2.imwrite(str(path2), img2)
    print(f"  -> Saved: {path2} ({len(hard_cache.hard_boxes)} hard region boxes)")

    # -------------------------------------------------------------------------
    # IMAGE 3: Ground Truth
    # -------------------------------------------------------------------------
    print("[main] Generating Image 3: Ground truth...")
    img3 = img.copy()
    draw_boxes(img3, gt_boxes, color=(0, 255, 0), classes=None, thickness=2)
    path3 = out_dir / f"3_ground_truth_{image_path.stem}.jpg"
    cv2.imwrite(str(path3), img3)
    print(f"  -> Saved: {path3} ({len(gt_boxes)} GT boxes)")

    # -------------------------------------------------------------------------
    # IMAGE 4 & 5: RL-SAHI Pipeline (Orange RL slices & Final merged detection)
    # -------------------------------------------------------------------------
    print("[main] Running Adaptive RL-SAHI inference pipeline...")
    temp_infer_dir = out_dir / "temp_pipeline"
    meta = inferencer.infer_image(image_path, out_dir=temp_infer_dir, use_cache=False)
    
    # Image 4: RL Slices in orange color (BGR: 0, 165, 255)
    print("[main] Generating Image 4: RL slices...")
    slice_rois = np.array(
        [s["roi"] for s in meta.get("slices", []) if "roi" in s], dtype=np.float32
    ).reshape(-1, 4) if meta.get("slices") else np.zeros((0, 4), dtype=np.float32)

    img4 = img.copy()
    draw_boxes(img4, slice_rois, color=(0, 165, 255), classes=None, thickness=5)
    path4 = out_dir / f"4_rl_slices_orange_{image_path.stem}.jpg"
    cv2.imwrite(str(path4), img4)
    print(f"  -> Saved: {path4} ({len(slice_rois)} slice ROIs)")

    # Image 5: Final merged detection
    print("[main] Generating Image 5: Final detection...")
    pred_path = Path(meta["prediction_file"]) if meta.get("prediction_file") else temp_infer_dir / "detections" / f"{image_path.stem}.txt"
    final_boxes, final_scores, final_classes, final_sources = _load_prediction_txt(pred_path)

    img5 = img.copy()
    if len(final_boxes) > 0:
        mask_full = final_sources == 0
        draw_boxes(img5, final_boxes[mask_full], color=(0, 255, 0), classes=None, thickness=2)
        draw_boxes(img5, final_boxes[~mask_full], color=(0, 165, 255), classes=None, thickness=2)
    path5 = out_dir / f"5_final_detection_{image_path.stem}.jpg"
    cv2.imwrite(str(path5), img5)
    print(f"  -> Saved: {path5} ({len(final_boxes)} final detections)")

    # -------------------------------------------------------------------------
    # BONUS: Combined 2x3 Grid Summary Image
    # -------------------------------------------------------------------------
    print("[main] Creating combined 2x3 grid summary image...")
    h1 = add_header(img1, f"1. YOLO Detect Conf {args.conf} (Green)")
    h2 = add_header(img2, "2. Hard Regions (Red)")
    h3 = add_header(img3, "3. Ground Truth")
    h4 = add_header(img4, "4. RL Slices Explored (Orange)")
    h5 = add_header(img5, "5. Final RL-SAHI Detection")

    stats = [
        ("Image Name", image_path.name),
        ("Resolution", f"{img_w}x{img_h}"),
        (f"YOLO Conf>={args.conf} Boxes", str(len(boxes_thresh))),
        ("Hard Region Boxes", str(len(hard_cache.hard_boxes))),
        ("Ground Truth Boxes", str(len(gt_boxes))),
        ("RL Slices Explored", str(len(slice_rois))),
        ("Final Merged Detections", str(len(final_boxes))),
        ("Total Inference Time", f"{meta.get('timing', {}).get('total_ms', 0.0):.1f} ms"),
    ]
    info_panel = create_info_panel(img_w, h1.shape[0], stats)

    row1 = np.hstack([h1, h2, h3])
    row2 = np.hstack([h4, h5, info_panel])
    grid = np.vstack([row1, row2])
    grid_path = out_dir / f"summary_grid_5_images_{image_path.stem}.jpg"
    cv2.imwrite(str(grid_path), grid)
    print(f"  -> Saved Grid Summary: {grid_path}")
    print("\n[SUCCESS] All 5 images and summary grid generated successfully!")


if __name__ == "__main__":
    main()
