from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..core.types import Detections


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(image_root: Path, split: str, limit: int | None = None) -> list[Path]:
    split_dir = Path(image_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Image split does not exist: {split_dir}")
    images = sorted(
        path for path in split_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    return images if limit is None else images[: max(int(limit), 0)]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def label_path_for(image_path: Path, label_root: Path, split: str) -> Path:
    return Path(label_root) / split / f"{image_path.stem}.txt"


def load_yolo_labels(path: Path, image_shape: tuple[int, int]) -> Detections:
    if not path.exists():
        return Detections.empty()
    rows: list[list[float]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = raw_line.strip().split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"Invalid YOLO label at {path}:{line_number}")
        rows.append([float(value) for value in fields[:5]])
    if not rows:
        return Detections.empty()
    values = np.asarray(rows, dtype=np.float32)
    height, width = image_shape
    centers = values[:, 1:3] * np.asarray([width, height], dtype=np.float32)
    sizes = values[:, 3:5] * np.asarray([width, height], dtype=np.float32)
    boxes = np.concatenate([centers - sizes / 2.0, centers + sizes / 2.0], axis=1)
    return Detections(
        boxes=boxes,
        scores=np.ones((len(values),), dtype=np.float32),
        classes=values[:, 0].astype(np.int64),
    )
