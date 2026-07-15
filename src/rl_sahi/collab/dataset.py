from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml

from rl_sahi.common.box_transforms import xyxy_to_xywhn
from rl_sahi.common.data import ensure_dir, image_to_label_path, read_image, read_yolo_labels


@dataclass(slots=True)
class CollaborativeDatasetConfig:
    roi_source: str = "accepted"
    min_visibility: float = 0.5
    min_box_size: float = 2.0
    include_empty: bool = False
    target_classes: tuple[int, ...] | None = None
    image_ext: str = ".jpg"
    jpeg_quality: int = 95


@dataclass(slots=True)
class CollaborativeDatasetSummary:
    metadata_expected: int = 0
    metadata_files: int = 0
    missing_metadata: int = 0
    rois_seen: int = 0
    rois_selected: int = 0
    invalid_rois: int = 0
    empty_crops_skipped: int = 0
    crops_written: int = 0
    labels_written: int = 0
    max_class_id: int = -1

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def roi_to_crop_bounds(roi: Iterable[float], image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = image_shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in roi]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, w)
    y2 = min(y2, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def project_labels_to_crop(
    classes: np.ndarray,
    boxes: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
    *,
    min_visibility: float = 0.5,
    min_box_size: float = 2.0,
    target_classes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = crop_bounds
    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    classes = np.asarray(classes, dtype=np.float32).reshape(-1)
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(classes) == 0 or boxes.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    original_w = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
    original_h = np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    original_area = np.maximum(original_w * original_h, 1e-9)

    clipped = boxes.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(x1, x2)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(y1, y2)
    clipped_w = np.maximum(clipped[:, 2] - clipped[:, 0], 0.0)
    clipped_h = np.maximum(clipped[:, 3] - clipped[:, 1], 0.0)
    visibility = (clipped_w * clipped_h) / original_area

    keep = (
        (visibility >= float(min_visibility))
        & (clipped_w >= float(min_box_size))
        & (clipped_h >= float(min_box_size))
    )
    if target_classes is not None:
        target = np.asarray(target_classes, dtype=np.int64)
        keep &= np.isin(classes.astype(np.int64), target)

    if not np.any(keep):
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

    local_boxes = clipped[keep].copy()
    local_boxes[:, [0, 2]] -= float(x1)
    local_boxes[:, [1, 3]] -= float(y1)
    yolo_boxes = xyxy_to_xywhn(local_boxes, (crop_h, crop_w))
    return classes[keep].astype(np.float32), yolo_boxes.astype(np.float32)


def build_yolo_crop_dataset(
    *,
    metadata_dir: Path,
    image_root: Path,
    label_root: Path,
    out_dir: Path,
    split: str,
    image_paths: Iterable[Path] | None = None,
    cfg: CollaborativeDatasetConfig | None = None,
) -> CollaborativeDatasetSummary:
    cfg = cfg or CollaborativeDatasetConfig()
    if cfg.roi_source not in {"accepted", "rejected", "all"}:
        raise ValueError("roi_source must be one of: accepted, rejected, all")
    if not cfg.image_ext.startswith("."):
        raise ValueError("image_ext must start with '.'")

    metadata_dir = Path(metadata_dir)
    image_root = Path(image_root)
    label_root = Path(label_root)
    out_dir = Path(out_dir)
    for split_name in ("train", "val", "test"):
        ensure_dir(out_dir / "images" / split_name)
        ensure_dir(out_dir / "labels" / split_name)
    image_out_dir = ensure_dir(out_dir / "images" / split)
    label_out_dir = ensure_dir(out_dir / "labels" / split)

    expected_images = None if image_paths is None else [Path(path) for path in image_paths]
    metadata_inputs = (
        [(metadata_dir / f"{path.stem}.json", path) for path in expected_images]
        if expected_images is not None
        else [(path, None) for path in sorted(metadata_dir.glob("*.json"))]
    )

    summary = CollaborativeDatasetSummary(metadata_expected=len(metadata_inputs))
    manifest_rows: list[dict[str, Any]] = []

    for metadata_path, fallback_image_path in metadata_inputs:
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            summary.missing_metadata += 1
            continue

        summary.metadata_files += 1
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        image_path = _metadata_image_path(meta, fallback_image_path)
        image = read_image(image_path)
        image_shape = image.shape[:2]
        label_path = image_to_label_path(image_path, image_root, label_root)
        classes, boxes = read_yolo_labels(label_path, image_shape)

        selected_slices = _selected_slices(meta.get("slices", []), cfg.roi_source)
        summary.rois_seen += len(meta.get("slices", []))
        summary.rois_selected += len(selected_slices)

        for local_index, slice_meta in enumerate(selected_slices, start=1):
            roi = slice_meta.get("roi")
            if roi is None:
                summary.invalid_rois += 1
                continue
            crop_bounds = roi_to_crop_bounds(roi, image_shape)
            if crop_bounds is None:
                summary.invalid_rois += 1
                continue

            crop_classes, crop_labels = project_labels_to_crop(
                classes,
                boxes,
                crop_bounds,
                min_visibility=cfg.min_visibility,
                min_box_size=cfg.min_box_size,
                target_classes=cfg.target_classes,
            )
            if len(crop_classes) == 0 and not cfg.include_empty:
                summary.empty_crops_skipped += 1
                continue

            x1, y1, x2, y2 = crop_bounds
            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                summary.invalid_rois += 1
                continue

            crop_name = _crop_filename(image_path.stem, slice_meta, local_index, cfg.image_ext)
            crop_image_path = image_out_dir / crop_name
            crop_label_path = (label_out_dir / crop_name).with_suffix(".txt")
            _write_image(crop_image_path, crop, cfg)
            _write_yolo_label(crop_label_path, crop_classes, crop_labels)

            summary.crops_written += 1
            summary.labels_written += int(len(crop_classes))
            if len(crop_classes) > 0:
                summary.max_class_id = max(summary.max_class_id, int(np.max(crop_classes)))

            manifest_rows.append(
                {
                    "crop": str(crop_image_path.relative_to(out_dir)),
                    "label": str(crop_label_path.relative_to(out_dir)),
                    "source_image": str(image_path),
                    "source_label": str(label_path),
                    "metadata": str(metadata_path),
                    "accepted": bool(slice_meta.get("accepted", False)),
                    "attempt_index": slice_meta.get("attempt_index"),
                    "slice_index": slice_meta.get("slice_index"),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "labels": len(crop_classes),
                }
            )

    _write_manifest(out_dir / f"manifest_{split}.csv", manifest_rows)
    _write_data_yaml(out_dir, max(summary.max_class_id, _scan_max_class_id(out_dir / "labels")))
    return summary


def _metadata_image_path(meta: dict[str, Any], fallback_image_path: Path | None) -> Path:
    raw = meta.get("image")
    if raw:
        path = Path(str(raw))
        if path.exists():
            return path
    if fallback_image_path is not None:
        return Path(fallback_image_path)
    raise FileNotFoundError(f"Metadata does not point to an existing image: {raw!r}")


def _selected_slices(slices: list[dict[str, Any]], roi_source: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in slices:
        accepted = bool(item.get("accepted", False))
        if roi_source == "accepted" and not accepted:
            continue
        if roi_source == "rejected" and accepted:
            continue
        selected.append(item)
    return selected


def _crop_filename(stem: str, slice_meta: dict[str, Any], local_index: int, image_ext: str) -> str:
    attempt = slice_meta.get("attempt_index")
    slice_index = slice_meta.get("slice_index")
    accepted = "acc" if bool(slice_meta.get("accepted", False)) else "rej"
    attempt_part = f"a{int(attempt):03d}" if attempt is not None else f"i{local_index:03d}"
    slice_part = f"s{int(slice_index):03d}" if slice_index is not None else f"r{local_index:03d}"
    return f"{stem}_{accepted}_{attempt_part}_{slice_part}{image_ext.lower()}"


def _write_image(path: Path, image: np.ndarray, cfg: CollaborativeDatasetConfig) -> None:
    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)]
    if not cv2.imwrite(str(path), image, params):
        raise RuntimeError(f"Failed to write crop image: {path}")


def _write_yolo_label(path: Path, classes: np.ndarray, boxes_xywhn: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for cls, box in zip(classes, boxes_xywhn):
            cx, cy, bw, bh = [float(v) for v in box]
            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "crop",
        "label",
        "source_image",
        "source_label",
        "metadata",
        "accepted",
        "attempt_index",
        "slice_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "labels",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_data_yaml(out_dir: Path, max_class_id: int) -> None:
    nc = max(max_class_id + 1, 1)
    data = {
        "path": str(Path(out_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": nc,
        "names": [f"class_{idx}" for idx in range(nc)],
    }
    with (out_dir / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _scan_max_class_id(label_root: Path) -> int:
    max_class_id = -1
    for label_path in Path(label_root).rglob("*.txt"):
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    max_class_id = max(max_class_id, int(float(parts[0])))
                except ValueError:
                    continue
    return max_class_id
