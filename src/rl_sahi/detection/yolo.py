from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from rl_sahi.common.cache import DetectionCache
from rl_sahi.common.data import read_image_shape
from rl_sahi.common.device import DeviceLike, configure_torch_runtime, configure_ultralytics_for_device
from rl_sahi.detection.features import DetectAuxCollector, FeatureCollector


DEFAULT_AUX_GRID_SIZE = 16
DEFAULT_SPATIAL_FEATURE_CHANNELS = 4


def load_yolo(weights: Path, device: DeviceLike = None) -> YOLO:
    model = YOLO(str(weights))
    resolved_device = configure_torch_runtime(device)
    configure_ultralytics_for_device(resolved_device)
    model.to(resolved_device)
    return model


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
