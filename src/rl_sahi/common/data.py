from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .boxes import area, covered_area_by_boxes, xywhn_to_xyxy


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(image_root: Path, split: str | None = None, limit: int | None = None) -> list[Path]:
    root = Path(image_root)
    search_root = root / split if split else root
    images = sorted(p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit is not None:
        images = images[:limit]
    return images


def image_id(image_path: Path) -> str:
    return Path(image_path).stem


def image_to_label_path(image_path: Path, image_root: Path, label_root: Path) -> Path:
    image_path = Path(image_path)
    image_root = Path(image_root)
    label_root = Path(label_root)
    rel = image_path.relative_to(image_root)
    return (label_root / rel).with_suffix(".txt")


def image_to_annotation_path(image_path: Path, image_root: Path, annotation_root: Path) -> Path:
    image_path = Path(image_path)
    image_root = Path(image_root)
    annotation_root = Path(annotation_root)
    rel = image_path.relative_to(image_root)
    return (annotation_root / rel).with_suffix(".txt")


def read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image


def read_image_shape(image_path: Path) -> tuple[int, int]:
    image = read_image(image_path)
    h, w = image.shape[:2]
    return int(h), int(w)


def read_yolo_labels(label_path: Path, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    label_path = Path(label_path)
    if not label_path.exists():
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    rows: list[list[float]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            rows.append([float(x) for x in parts[:5]])
    if not rows:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    arr = np.asarray(rows, dtype=np.float32)
    classes = arr[:, 0]
    boxes = xywhn_to_xyxy(arr[:, 1:5], image_shape)
    return classes, boxes


def read_visdrone_det_annotations(
    annotation_path: Path,
    image_shape: tuple[int, int],
    ignore_overlap_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    annotation_path = Path(annotation_path)
    if not annotation_path.exists():
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )

    gt_rows: list[list[float]] = []
    ignored_region_rows: list[list[float]] = []
    ignored_object_rows: list[list[float]] = []
    with annotation_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = [part.strip() for part in line.replace(",", " ").split()]
            if len(parts) < 6:
                continue
            x, y, w, h = (float(value) for value in parts[:4])
            score = float(parts[4])
            category = int(float(parts[5]))
            box = [x, y, x + w, y + h]
            if category <= 0:
                ignored_region_rows.append(box)
                continue
            if score <= 0.0 or category > 10:
                ignored_object_rows.append(box)
                continue
            gt_rows.append([float(category - 1), *box])

    img_h, img_w = image_shape

    def clip(rows: list[list[float]], has_class: bool) -> np.ndarray:
        width = 5 if has_class else 4
        if not rows:
            return np.zeros((0, width), dtype=np.float32)
        arr = np.asarray(rows, dtype=np.float32).reshape(-1, width)
        offset = 1 if has_class else 0
        arr[:, offset + 0] = np.clip(arr[:, offset + 0], 0.0, float(img_w))
        arr[:, offset + 1] = np.clip(arr[:, offset + 1], 0.0, float(img_h))
        arr[:, offset + 2] = np.clip(arr[:, offset + 2], 0.0, float(img_w))
        arr[:, offset + 3] = np.clip(arr[:, offset + 3], 0.0, float(img_h))
        valid = (arr[:, offset + 2] > arr[:, offset + 0]) & (arr[:, offset + 3] > arr[:, offset + 1])
        return arr[valid]

    gt = clip(gt_rows, has_class=True)
    ignored_regions = clip(ignored_region_rows, has_class=False)
    ignored_objects = clip(ignored_object_rows, has_class=False)
    if len(gt) == 0:
        classes = np.zeros((0,), dtype=np.float32)
        boxes = np.zeros((0, 4), dtype=np.float32)
    else:
        classes = gt[:, 0].astype(np.float32)
        boxes = gt[:, 1:5].astype(np.float32)

    if len(boxes) > 0 and len(ignored_regions) > 0:
        box_area = np.maximum(area(boxes), 1e-7)
        ignored_ratio = covered_area_by_boxes(boxes, ignored_regions) / box_area
        keep = ignored_ratio < float(ignore_overlap_threshold)
        classes = classes[keep]
        boxes = boxes[keep]

    ignored_parts = [part for part in (ignored_regions, ignored_objects) if len(part) > 0]
    ignored = (
        np.concatenate(ignored_parts, axis=0).astype(np.float32)
        if ignored_parts
        else np.zeros((0, 4), dtype=np.float32)
    )
    return classes, boxes, ignored.astype(np.float32)


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
