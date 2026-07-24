from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.hard_regions import HardRegionData
from ..core.types import Detections


def file_signature(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def detector_signature(
    weights: Path,
    imgsz: int,
    confidence: float,
    iou: float,
    max_detections: int,
    target_classes: tuple[int, ...],
) -> str:
    payload = {
        "weights": file_signature(weights),
        "imgsz": int(imgsz),
        "confidence": float(confidence),
        "iou": float(iou),
        "max_detections": int(max_detections),
        "target_classes": list(target_classes),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class DetectionDiskCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def full_path(self, split: str, image_path: Path) -> Path:
        return self.root / "detections" / split / f"{image_path.stem}.npz"

    def crop_path(self, split: str, image_path: Path, roi: np.ndarray) -> Path:
        coordinates = "_".join(str(int(round(float(value)))) for value in roi)
        key = hashlib.sha1(coordinates.encode("ascii")).hexdigest()[:16]
        return (
            self.root
            / "detections"
            / "crops"
            / split
            / image_path.stem
            / f"{key}.npz"
        )

    def hard_region_path(self, split: str, image_path: Path) -> Path:
        return self.root / "hard_regions" / split / f"{image_path.stem}.npz"

    def load(self, path: Path, signature: str) -> Detections | None:
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["signature"].item()) != signature:
                    return None
                return Detections(data["boxes"], data["scores"], data["classes"])
        except (OSError, KeyError, ValueError):
            return None

    def save(self, path: Path, signature: str, detections: Detections) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                signature=np.asarray(signature),
                boxes=detections.boxes,
                scores=detections.scores,
                classes=detections.classes,
            )
        temporary.replace(path)

    def load_hard_regions(
        self,
        path: Path,
        signature: str,
    ) -> HardRegionData | None:
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                if str(data["signature"].item()) != signature:
                    return None
                return HardRegionData(
                    gt_boxes=data["gt_boxes"],
                    gt_classes=data["gt_classes"],
                    hard_mask=data["hard_mask"],
                    best_iou=data["best_iou"],
                    best_score=data["best_score"],
                )
        except (OSError, KeyError, ValueError):
            return None

    def save_hard_regions(
        self,
        path: Path,
        signature: str,
        regions: HardRegionData,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                signature=np.asarray(signature),
                gt_boxes=regions.gt_boxes,
                gt_classes=regions.gt_classes,
                hard_mask=regions.hard_mask,
                best_iou=regions.best_iou,
                best_score=regions.best_score,
            )
        temporary.replace(path)
