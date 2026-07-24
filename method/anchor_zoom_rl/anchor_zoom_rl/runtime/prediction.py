from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import MethodConfig
from ..core.hard_regions import HardRegionData, build_hard_regions
from ..core.postprocess import translate_crop_detections
from ..core.types import Detections
from .cache import DetectionDiskCache, detector_signature, file_signature
from .detector import DetectorPair


class DetectionRunner:
    def __init__(self, cfg: MethodConfig, detectors: DetectorPair | None = None) -> None:
        self.cfg = cfg
        self.detectors = detectors or DetectorPair(
            cfg.paths.weights, cfg.paths.crop_weights, cfg.detector
        )
        self.cache = DetectionDiskCache(cfg.paths.cache_dir)
        self.full_signature = detector_signature(
            cfg.paths.weights,
            cfg.detector.full_imgsz,
            cfg.detector.full_confidence,
            cfg.detector.yolo_iou,
            cfg.detector.max_detections,
            cfg.detector.target_classes,
        )
        self.crop_signature = detector_signature(
            cfg.paths.crop_weights,
            cfg.detector.crop_imgsz,
            cfg.detector.crop_confidence,
            cfg.detector.yolo_iou,
            cfg.detector.max_detections,
            cfg.detector.target_classes,
        )

    def full(
        self,
        image: np.ndarray,
        image_path: Path,
        split: str,
        use_cache: bool,
    ) -> tuple[Detections, float, bool]:
        cache_path = self.cache.full_path(split, image_path)
        signature = _image_bound_signature(self.full_signature, image_path)
        if use_cache:
            cached = self.cache.load(cache_path, signature)
            if cached is not None:
                return cached, 0.0, True
        detections, elapsed_ms = self.detectors.predict_full(image)
        if use_cache:
            self.cache.save(cache_path, signature, detections)
        return detections, elapsed_ms, False

    def crop(
        self,
        image: np.ndarray,
        image_path: Path,
        split: str,
        roi: np.ndarray,
        use_cache: bool,
    ) -> tuple[Detections, float, bool]:
        rounded = _integer_roi(roi, image.shape[:2])
        cache_path = self.cache.crop_path(split, image_path, rounded)
        signature = _image_bound_signature(self.crop_signature, image_path)
        local = self.cache.load(cache_path, signature) if use_cache else None
        cache_hit = local is not None
        elapsed_ms = 0.0
        if local is None:
            x1, y1, x2, y2 = rounded.tolist()
            crop_image = image[y1:y2, x1:x2]
            if crop_image.size == 0:
                local = Detections.empty()
            else:
                local, elapsed_ms = self.detectors.predict_crop(crop_image)
            if use_cache:
                self.cache.save(cache_path, signature, local)
        return translate_crop_detections(local, rounded), float(elapsed_ms), cache_hit

    def hard_regions(
        self,
        full_detections: Detections,
        ground_truth: Detections,
        image_path: Path,
        label_path: Path,
        split: str,
        use_cache: bool,
    ) -> tuple[HardRegionData, bool]:
        path = self.cache.hard_region_path(split, image_path)
        signature = _hard_region_signature(
            self.full_signature,
            image_path,
            label_path,
            self.cfg.reward.match_iou,
            self.cfg.reward.hard_low_confidence,
        )
        if use_cache:
            cached = self.cache.load_hard_regions(path, signature)
            if cached is not None:
                return cached, True
        regions = build_hard_regions(
            full_detections,
            ground_truth,
            match_iou=self.cfg.reward.match_iou,
            low_confidence=self.cfg.reward.hard_low_confidence,
        )
        if use_cache:
            self.cache.save_hard_regions(path, signature, regions)
        return regions, False


def _integer_roi(roi: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    values = np.asarray(
        [
            np.floor(float(roi[0])),
            np.floor(float(roi[1])),
            np.ceil(float(roi[2])),
            np.ceil(float(roi[3])),
        ],
        dtype=np.int32,
    )
    values[[0, 2]] = np.clip(values[[0, 2]], 0, width)
    values[[1, 3]] = np.clip(values[[1, 3]], 0, height)
    values[2] = max(values[2], values[0] + 1)
    values[3] = max(values[3], values[1] + 1)
    return values


def _image_bound_signature(detector_hash: str, image_path: Path) -> str:
    payload = {
        "detector": detector_hash,
        "image": file_signature(image_path),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _hard_region_signature(
    detector_hash: str,
    image_path: Path,
    label_path: Path,
    match_iou: float,
    low_confidence: float,
) -> str:
    payload = {
        "detector": detector_hash,
        "image": file_signature(image_path),
        "label": file_signature(label_path),
        "match_iou": float(match_iou),
        "low_confidence": float(low_confidence),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
