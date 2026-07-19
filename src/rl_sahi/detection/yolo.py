from __future__ import annotations

import time
from pathlib import Path

import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO

from rl_sahi.common.cache import DetectionCache
from rl_sahi.common.data import read_image_shape
from rl_sahi.common.device import DeviceLike, configure_torch_runtime, configure_ultralytics_for_device
from rl_sahi.detection.features import DetectAuxCollector, FeatureCollector


DEFAULT_AUX_GRID_SIZE = 16
DEFAULT_SPATIAL_FEATURE_CHANNELS = 4


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.xyxy)

class _FakeResults:
    def __init__(self, boxes, speed):
        self.boxes = boxes
        self.speed = speed

class _DummyModel:
    def __init__(self):
        self.model = []

class LegacyYOLOWrapper:
    def __init__(self, weights: Path, device: DeviceLike = None):
        self.weights = Path(weights).resolve()
        
        # Add tph-yolov5 to path
        repo_dir = self.weights.parent / "tph-yolov5"
        if not repo_dir.exists():
            raise FileNotFoundError(f"tph-yolov5 repository not found at {repo_dir}")
        sys.path.insert(0, str(repo_dir))
        try:
            from models.experimental import attempt_load
            self.pytorch_model = attempt_load(str(self.weights), map_location="cpu")
        finally:
            sys.path.pop(0)
            
        self.device = configure_torch_runtime(device)
        self.pytorch_model.to(self.device)
        self.pytorch_model.eval()
        self.stride = int(self.pytorch_model.stride.max())

    @property
    def model(self):
        return _DummyModel()

    def to(self, device):
        self.device = configure_torch_runtime(device)
        self.pytorch_model.to(self.device)

    def predict(
        self,
        source,
        imgsz=640,
        conf=0.25,
        iou=0.45,
        max_det=3000,
        batch=1,
        device=None,
        verbose=False,
    ):
        if device is not None:
            self.to(device)
            
        repo_dir = self.weights.parent / "tph-yolov5"
        sys.path.insert(0, str(repo_dir))
        try:
            from utils.augmentations import letterbox
            from utils.general import non_max_suppression, scale_coords
        finally:
            sys.path.pop(0)

        is_single = not isinstance(source, list)
        sources = [source] if is_single else source

        images_rgb = []
        original_shapes = []
        
        preprocess_start = time.perf_counter()
        for s in sources:
            if isinstance(s, (str, Path)):
                img0 = cv2.imread(str(s))
            elif isinstance(s, np.ndarray):
                img0 = s
            else:
                raise TypeError(f"Unsupported source type: {type(s)}")
            original_shapes.append(img0.shape[:2])
            # auto=True for single image (rectangular inference, matches training);
            # auto=False for batch (all images must be same padded size).
            img = letterbox(img0, imgsz, stride=self.stride, auto=(len(sources) == 1))[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            images_rgb.append(img)
            
        tensor = torch.from_numpy(np.stack(images_rgb)).to(self.device).float() / 255.0
        preprocess_time = (time.perf_counter() - preprocess_start) * 1000

        inference_start = time.perf_counter()
        with torch.no_grad():
            pred, _ = self.pytorch_model(tensor)
        inference_time = (time.perf_counter() - inference_start) * 1000

        postprocess_start = time.perf_counter()
        pred = non_max_suppression(pred, conf, iou, max_det=max_det)
        
        results = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], original_shapes[i]).round()
                
            xyxy = det[:, :4]
            scores = det[:, 4]
            classes = det[:, 5]
            
            speed = {
                "preprocess": preprocess_time / len(sources),
                "inference": inference_time / len(sources),
                "postprocess": (time.perf_counter() - postprocess_start) * 1000 / len(sources)
            }
            results.append(_FakeResults(_FakeBoxes(xyxy, scores, classes), speed))
            
        return results

def load_yolo(weights: Path, device: DeviceLike = None):
    try:
        model = YOLO(str(weights))
        resolved_device = configure_torch_runtime(device)
        configure_ultralytics_for_device(resolved_device)
        model.to(resolved_device)
        return model
    except Exception as e:
        if "NOT forwards compatible" in str(e) or "originally trained with" in str(e):
            return LegacyYOLOWrapper(weights, device)
        raise


def detect_one_image(
    model: YOLO,
    image_path: Path,
    imgsz: int = 640,
    conf: float = 0.01,
    iou: float = 0.7,
    max_det: int = 3000,
    device: DeviceLike = None,
    feature_layers: tuple[int, ...] = (10,),
    aux_grid_size: int = DEFAULT_AUX_GRID_SIZE,
    spatial_feature_channels: int = DEFAULT_SPATIAL_FEATURE_CHANNELS,
    timing: dict[str, float] | None = None,
    source_image: np.ndarray | None = None,
) -> DetectionCache:
    if source_image is None:
        shape_start = time.perf_counter()
        image_shape = read_image_shape(image_path)
        _add_timing(timing, "initial_image_read_ms", _elapsed_ms(shape_start))
        predict_source = str(image_path)
    else:
        image_shape = (int(source_image.shape[0]), int(source_image.shape[1]))
        predict_source = source_image
    resolved_device = configure_torch_runtime(device)
    configure_ultralytics_for_device(resolved_device)
    with FeatureCollector(model, feature_layers) as collector, DetectAuxCollector(model) as aux_collector:
        collector.clear()
        aux_collector.clear()
        predict_start = time.perf_counter()
        results = model.predict(
            source=predict_source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            device=resolved_device,
            verbose=False,
        )
        _add_timing(timing, "initial_yolo_wall_ms", _elapsed_ms(predict_start))
        _add_result_speeds(timing, results)
        feature_start = time.perf_counter()
        feature = collector.vector()
        objectness_map, spatial_feature_map = aux_collector.maps(
            grid_size=aux_grid_size,
            spatial_feature_channels=spatial_feature_channels,
        )
        _add_timing(timing, "initial_feature_extract_ms", _elapsed_ms(feature_start))
    result = results[0]
    transfer_start = time.perf_counter()
    if result.boxes is None or len(result.boxes) == 0:
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.float32)
    else:
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.float32)
    _add_timing(timing, "initial_result_transfer_ms", _elapsed_ms(transfer_start))
    return DetectionCache(
        image_path=str(image_path),
        image_shape=image_shape,
        boxes=boxes,
        scores=scores,
        classes=classes,
        feature=feature,
        feature_layers=feature_layers,
        objectness_map=objectness_map,
        spatial_feature_map=spatial_feature_map,
    )


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _add_timing(timing: dict[str, float] | None, key: str, value: float) -> None:
    if timing is not None:
        timing[key] = float(timing.get(key, 0.0)) + float(value)


def _add_result_speeds(timing: dict[str, float] | None, results) -> None:
    if timing is None:
        return
    key_map = {
        "preprocess": "initial_preprocess_ms",
        "inference": "initial_yolo_inference_ms",
        "postprocess": "initial_postprocess_ms",
    }
    for result in results:
        speed = getattr(result, "speed", None) or {}
        for speed_key, timing_key in key_map.items():
            _add_timing(timing, timing_key, float(speed.get(speed_key, 0.0)))
