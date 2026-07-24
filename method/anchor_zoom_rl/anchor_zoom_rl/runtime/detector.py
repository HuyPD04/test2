from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from ..config import DetectorConfig
from ..core.types import Detections


class YoloDetector:
    def __init__(self, weights: Path, cfg: DetectorConfig, crop: bool = False) -> None:
        from ultralytics import YOLO

        self.weights = Path(weights).resolve()
        self.cfg = cfg
        self.crop = bool(crop)
        self.device = cfg.device
        self.model = YOLO(str(self.weights))

    @property
    def imgsz(self) -> int:
        return self.cfg.crop_imgsz if self.crop else self.cfg.full_imgsz

    @property
    def confidence(self) -> float:
        return self.cfg.crop_confidence if self.crop else self.cfg.full_confidence

    def predict(self, image: np.ndarray | str | Path) -> tuple[Detections, float]:
        source = str(image) if isinstance(image, Path) else image
        start = time.perf_counter()
        results = self.model.predict(
            source=source,
            imgsz=self.imgsz,
            conf=self.confidence,
            iou=self.cfg.yolo_iou,
            max_det=self.cfg.max_detections,
            classes=list(self.cfg.target_classes) if self.cfg.target_classes else None,
            device=self.device,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return Detections.empty(), float(elapsed_ms)
        detections = Detections(
            boxes.xyxy.detach().cpu().numpy(),
            boxes.conf.detach().cpu().numpy(),
            boxes.cls.detach().cpu().numpy().astype(np.int64),
        )
        return detections.filter_classes(self.cfg.target_classes), float(elapsed_ms)


class DetectorPair:
    def __init__(self, full_weights: Path, crop_weights: Path, cfg: DetectorConfig) -> None:
        if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
            print("[device] CUDA is unavailable; detector is falling back to CPU.")
            cfg.device = "cpu"
        self.full = YoloDetector(full_weights, cfg, crop=False)
        if Path(full_weights).resolve() == Path(crop_weights).resolve():
            self.crop = self.full
            self.crop.crop = False
            self._shared = True
        else:
            self.crop = YoloDetector(crop_weights, cfg, crop=True)
            self._shared = False
        self.cfg = cfg

    def predict_full(self, image: np.ndarray) -> tuple[Detections, float]:
        if self._shared:
            self.full.crop = False
        return self.full.predict(image)

    def predict_crop(self, crop: np.ndarray) -> tuple[Detections, float]:
        if self._shared:
            self.full.crop = True
            try:
                return self.full.predict(crop)
            finally:
                self.full.crop = False
        return self.crop.predict(crop)
