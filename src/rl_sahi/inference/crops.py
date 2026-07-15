from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from rl_sahi.common.device import DeviceLike, configure_torch_runtime, configure_ultralytics_for_device


def crop_roi(image_path: Path, roi: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return crop_array(image, roi)


def crop_array(image: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def run_yolo_on_crop(
    model: YOLO,
    image_path: Path,
    roi: np.ndarray,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: DeviceLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return run_yolo_on_crops(
        model,
        [image_path],
        [roi],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        device=device,
    )[0]


def run_yolo_on_crops(
    model: YOLO,
    image_paths: list[Path],
    rois: list[np.ndarray],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    device: DeviceLike,
    timing: dict[str, float] | None = None,
    source_image: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if len(image_paths) != len(rois):
        raise ValueError("image_paths and rois must have the same length")
    empty = (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )
    outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [empty for _ in image_paths]
    crops: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    output_indices: list[int] = []
    unique_paths = {Path(path) for path in image_paths}
    if source_image is not None and len(unique_paths) > 1:
        raise ValueError("source_image can only be reused when all image_paths are the same")
    image_cache: dict[Path, np.ndarray] = (
        {next(iter(unique_paths)): source_image} if source_image is not None and unique_paths else {}
    )
    for index, (image_path, roi) in enumerate(zip(image_paths, rois)):
        image_path = Path(image_path)
        image = image_cache.get(image_path)
        if image is None:
            read_start = time.perf_counter()
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            _add_timing(timing, "crop_image_read_ms", _elapsed_ms(read_start))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")
            image_cache[image_path] = image
        extract_start = time.perf_counter()
        crop, offset = crop_array(image, roi)
        _add_timing(timing, "crop_extract_ms", _elapsed_ms(extract_start))
        if crop.size == 0:
            continue
        crops.append(crop)
        offsets.append(offset)
        output_indices.append(index)

    if not crops:
        return outputs

    resolved_device = configure_torch_runtime(device)
    configure_ultralytics_for_device(resolved_device)
    predict_start = time.perf_counter()
    results = model.predict(
        crops,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        batch=len(crops),
        device=resolved_device,
        verbose=False,
    )
    _add_timing(timing, "crop_yolo_wall_ms", _elapsed_ms(predict_start))
    _add_result_speeds(timing, results, "crop")
    transfer_start = time.perf_counter()
    for output_index, offset, result in zip(output_indices, offsets, results):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        boxes[:, [0, 2]] += offset[0]
        boxes[:, [1, 3]] += offset[1]
        scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.float32)
        outputs[output_index] = (boxes, scores, classes)
    _add_timing(timing, "crop_result_transfer_ms", _elapsed_ms(transfer_start))
    return outputs


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _add_timing(timing: dict[str, float] | None, key: str, value: float) -> None:
    if timing is not None:
        timing[key] = float(timing.get(key, 0.0)) + float(value)


def _add_result_speeds(timing: dict[str, float] | None, results, prefix: str) -> None:
    if timing is None:
        return
    key_map = {
        "preprocess": f"{prefix}_preprocess_ms",
        "inference": f"{prefix}_yolo_inference_ms",
        "postprocess": f"{prefix}_postprocess_ms",
    }
    for result in results:
        speed = getattr(result, "speed", None) or {}
        for speed_key, timing_key in key_map.items():
            _add_timing(timing, timing_key, float(speed.get(speed_key, 0.0)))
