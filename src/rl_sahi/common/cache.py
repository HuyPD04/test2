from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .data import image_id


DETECTION_CACHE_VERSION = 4
HARD_REGION_CACHE_VERSION = 1


@dataclass(slots=True)
class DetectionCache:
    image_path: str
    image_shape: tuple[int, int]
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    feature: np.ndarray
    feature_layers: tuple[int, ...]
    objectness_map: np.ndarray
    spatial_feature_map: np.ndarray
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class HardRegionCache:
    image_path: str
    image_shape: tuple[int, int]
    hard_boxes: np.ndarray
    small_gt_boxes: np.ndarray
    gt_boxes: np.ndarray
    matched_iou: np.ndarray
    matched_score: np.ndarray
    metadata: dict[str, Any] | None = None


def detection_cache_path(cache_root: Path, split: str, image_path: Path) -> Path:
    return Path(cache_root) / "detections" / split / f"{image_id(image_path)}.npz"


def hard_region_cache_path(cache_root: Path, split: str, image_path: Path) -> Path:
    return Path(cache_root) / "hard_regions" / split / f"{image_id(image_path)}.npz"


def _normalize_metadata(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _normalize_metadata(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata(item) for item in value]
    return value


def _metadata_json(metadata: dict[str, Any] | None) -> str:
    return json.dumps(_normalize_metadata(metadata or {}), sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=32)
def _file_sha256(path_str: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path_str).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    path = Path(path)
    fingerprint: dict[str, Any] = {"path": str(path.resolve())}
    if not path.exists():
        fingerprint["exists"] = False
        return fingerprint
    stat = path.stat()
    fingerprint["exists"] = True
    fingerprint["size"] = int(stat.st_size)
    fingerprint["mtime_ns"] = int(stat.st_mtime_ns)
    fingerprint["sha256"] = _file_sha256(str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    return fingerprint


def detection_cache_metadata(
    weights: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    feature_layers: tuple[int, ...],
    aux_grid_size: int,
    spatial_feature_channels: int,
) -> dict[str, Any]:
    return {
        "weights": file_fingerprint(Path(weights)),
        "imgsz": int(imgsz),
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "feature_layers": tuple(int(x) for x in feature_layers),
        "aux_grid_size": int(aux_grid_size),
        "spatial_feature_channels": int(spatial_feature_channels),
    }


def save_detection_cache(path: Path, cache: DetectionCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.asarray(DETECTION_CACHE_VERSION, dtype=np.int32),
        metadata_json=np.asarray(_metadata_json(cache.metadata)),
        image_path=np.asarray(cache.image_path),
        image_shape=np.asarray(cache.image_shape, dtype=np.int32),
        boxes=cache.boxes.astype(np.float32),
        scores=cache.scores.astype(np.float32),
        classes=cache.classes.astype(np.float32),
        feature=cache.feature.astype(np.float32),
        feature_layers=np.asarray(cache.feature_layers, dtype=np.int32),
        objectness_map=cache.objectness_map.astype(np.float32),
        spatial_feature_map=cache.spatial_feature_map.astype(np.float32),
    )


def hard_region_cache_metadata(
    detection_metadata: dict[str, Any] | None,
    small_area_ratio: float,
    small_area_percentile: float | None,
    match_iou: float,
    min_detect_score: float,
    target_classes: tuple[int, ...],
    class_mapping: Any,
) -> dict[str, Any]:
    return {
        "detection": detection_metadata or {},
        "small_area_ratio": float(small_area_ratio),
        "small_area_percentile": None if small_area_percentile is None else float(small_area_percentile),
        "match_iou": float(match_iou),
        "min_detect_score": float(min_detect_score),
        "target_classes": tuple(int(x) for x in target_classes),
        "class_mapping": {
            "model_to_label": dict(getattr(class_mapping, "model_to_label", {}) or {}),
            "label_to_eval": dict(getattr(class_mapping, "label_to_eval", {}) or {}),
        },
    }


def _comparison_metadata(value: Any) -> Any:
    value = _normalize_metadata(value)
    if isinstance(value, dict):
        normalized = {key: _comparison_metadata(item) for key, item in value.items()}
        if normalized.get("exists") is True and "sha256" in normalized:
            normalized.pop("path", None)
            normalized.pop("mtime_ns", None)
        return normalized
    if isinstance(value, list):
        return [_comparison_metadata(item) for item in value]
    return value


def _compare_metadata(actual_json: str, expected_dict: dict[str, Any]) -> bool:
    try:
        actual = json.loads(actual_json)
        return _comparison_metadata(actual) == _comparison_metadata(expected_dict)
    except Exception:
        return False


def metadata_matches(actual: dict[str, Any] | None, expected: dict[str, Any] | None) -> bool:
    return _comparison_metadata(actual or {}) == _comparison_metadata(expected or {})


def detection_cache_is_current(path: Path, expected_metadata: dict[str, Any] | None = None) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        if "cache_version" not in data.files:
            return False
        version = int(np.asarray(data["cache_version"]).item())
        return (
            version >= DETECTION_CACHE_VERSION
            and "metadata_json" in data.files
            and "objectness_map" in data.files
            and "spatial_feature_map" in data.files
            and (
                expected_metadata is None
                or _compare_metadata(str(data["metadata_json"].item()), expected_metadata)
            )
        )


def load_detection_cache(path: Path) -> DetectionCache:
    with np.load(path, allow_pickle=False) as data:
        shape = data["image_shape"].astype(np.int32).tolist()
        objectness_map = (
            data["objectness_map"].astype(np.float32)
            if "objectness_map" in data.files
            else np.zeros((0,), dtype=np.float32)
        )
        spatial_feature_map = (
            data["spatial_feature_map"].astype(np.float32)
            if "spatial_feature_map" in data.files
            else np.zeros((0,), dtype=np.float32)
        )
        return DetectionCache(
            image_path=str(data["image_path"].item()),
            image_shape=(int(shape[0]), int(shape[1])),
            boxes=data["boxes"].astype(np.float32),
            scores=data["scores"].astype(np.float32),
            classes=data["classes"].astype(np.float32),
            feature=data["feature"].astype(np.float32),
            feature_layers=tuple(int(x) for x in data["feature_layers"].tolist()),
            objectness_map=objectness_map,
            spatial_feature_map=spatial_feature_map,
            metadata=(
                json.loads(str(data["metadata_json"].item()))
                if "metadata_json" in data.files
                else {}
            ),
        )


def save_hard_region_cache(path: Path, cache: HardRegionCache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_version=np.asarray(HARD_REGION_CACHE_VERSION, dtype=np.int32),
        metadata_json=np.asarray(_metadata_json(cache.metadata)),
        image_path=np.asarray(cache.image_path),
        image_shape=np.asarray(cache.image_shape, dtype=np.int32),
        hard_boxes=cache.hard_boxes.astype(np.float32),
        small_gt_boxes=cache.small_gt_boxes.astype(np.float32),
        gt_boxes=cache.gt_boxes.astype(np.float32),
        matched_iou=cache.matched_iou.astype(np.float32),
        matched_score=cache.matched_score.astype(np.float32),
    )


def hard_region_cache_is_current(path: Path, expected_metadata: dict[str, Any] | None = None) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as data:
        if "cache_version" not in data.files or "metadata_json" not in data.files:
            return False
        version = int(np.asarray(data["cache_version"]).item())
        return version >= HARD_REGION_CACHE_VERSION and (
            expected_metadata is None
            or _compare_metadata(str(data["metadata_json"].item()), expected_metadata)
        )


def load_hard_region_cache(path: Path) -> HardRegionCache:
    with np.load(path, allow_pickle=False) as data:
        shape = data["image_shape"].astype(np.int32).tolist()
        return HardRegionCache(
            image_path=str(data["image_path"].item()),
            image_shape=(int(shape[0]), int(shape[1])),
            hard_boxes=data["hard_boxes"].astype(np.float32),
            small_gt_boxes=data["small_gt_boxes"].astype(np.float32),
            gt_boxes=data["gt_boxes"].astype(np.float32),
            matched_iou=data["matched_iou"].astype(np.float32),
            matched_score=data["matched_score"].astype(np.float32),
            metadata=(json.loads(str(data["metadata_json"].item())) if "metadata_json" in data.files else {}),
        )
