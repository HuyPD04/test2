from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import iter_images
from rl_sahi.common.device import print_device_info, resolve_torch_device
from rl_sahi.inference.config import InferenceConfig
from rl_sahi.inference.pipeline import AdaptiveSahiInferencer


TIMING_KEYS = (
    "total_ms",
    "image_read_ms",
    "initial_detection_ms",
    "initial_yolo_wall_ms",
    "initial_preprocess_ms",
    "initial_yolo_inference_ms",
    "initial_postprocess_ms",
    "initial_feature_extract_ms",
    "initial_result_transfer_ms",
    "rollout_ms",
    "rollout_static_ms",
    "rollout_env_init_ms",
    "rollout_state_ms",
    "rollout_valid_ms",
    "rollout_policy_ms",
    "rollout_step_ms",
    "roi_prefilter_ms",
    "candidate_evaluation_ms",
    "crop_inference_ms",
    "crop_image_read_ms",
    "crop_extract_ms",
    "crop_yolo_wall_ms",
    "crop_preprocess_ms",
    "crop_yolo_inference_ms",
    "crop_postprocess_ms",
    "crop_result_transfer_ms",
    "merge_ms",
)


def _bool_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _optional_float(value, default: float | None = None) -> float | None:
    raw = default if value is None else value
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null", "false", "off"}:
        return None
    return float(raw)


def _optional_path(cfg, name: str) -> Path | None:
    try:
        return cfg.path_value(name)
    except KeyError:
        return None


def _resolve_cli_path(value: Path, root: Path) -> Path:
    return value if value.is_absolute() else root / value


def _build_inference_config(project_cfg, device: str | None, policy_device: str | None) -> InferenceConfig:
    infer_cfg = project_cfg.section("infer")
    return InferenceConfig(
        full_imgsz=int(infer_cfg["full_imgsz"]),
        slice_imgsz=int(infer_cfg["slice_imgsz"]),
        full_conf=float(infer_cfg["full_conf"]),
        output_conf=float(infer_cfg["output_conf"]),
        iou=float(infer_cfg["iou"]),
        merge_iou=float(infer_cfg["merge_iou"]),
        max_det=int(infer_cfg["max_det"]),
        device=device,
        policy_device=policy_device,
        feature_layers=project_cfg.feature_layers("infer"),
        spatial_feature_layers=project_cfg.spatial_feature_layers("infer"),
        min_slice_detections=int(infer_cfg.get("min_slice_detections", 1)),
        min_slice_utility=float(infer_cfg.get("min_slice_utility", 0.5)),
        min_new_detection_score=float(infer_cfg.get("min_new_detection_score", 0.45)),
        duplicate_iou=float(infer_cfg.get("duplicate_iou", infer_cfg.get("merge_iou", 0.5))),
        boundary_margin=float(infer_cfg.get("boundary_margin", 2.0)),
        append_novel_only=_bool_value(infer_cfg.get("append_novel_only", False)),
        cross_class_duplicate_iou=_optional_float(infer_cfg.get("cross_class_duplicate_iou"), 0.85),
        cross_class_duplicate_ios=_optional_float(infer_cfg.get("cross_class_duplicate_ios"), 0.95),
        max_slice_attempts=int(infer_cfg.get("max_slice_attempts", 0)),
        roi_prefilter_enabled=_bool_value(infer_cfg.get("roi_prefilter_enabled", False)),
        roi_prefilter_topk=int(infer_cfg.get("roi_prefilter_topk", 3)),
        crop_batch_size=int(infer_cfg.get("crop_batch_size", 1)),
        max_consecutive_rejections=int(infer_cfg.get("max_consecutive_rejections", 0)),
        target_classes=project_cfg.target_classes(),
        require_stop_for_acceptance=_bool_value(infer_cfg.get("require_stop_for_acceptance", True)),
        # Output I/O and cache I/O must not contaminate an end-to-end speed result.
        save_predictions=False,
        save_metadata=False,
        save_visualization=False,
        batched_inference=_bool_value(infer_cfg.get("batched_inference", False)),
        use_wbf=_bool_value(infer_cfg.get("use_wbf", False)),
        nms_type=str(infer_cfg.get("nms_type", "standard")),
        gate_nms_type=str(infer_cfg.get("gate_nms_type", "standard")),
        class_mapping=ClassMapping.from_config(project_cfg.section("classes")),
    )


def _select_images(
    image_root: Path,
    split: str,
    image: Path | None,
    limit: int,
) -> list[Path]:
    if image is not None:
        if not image.exists():
            raise FileNotFoundError(f"Image not found: {image}")
        return [image]
    image_limit = None if limit < 0 else limit
    images = iter_images(image_root, split=split, limit=image_limit)
    if not images:
        raise FileNotFoundError(f"No images found for split '{split}'")
    return images


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


def _stats(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "median": _percentile(array, 50),
        "p90": _percentile(array, 90),
        "p95": _percentile(array, 95),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "index",
        "image",
        "is_warmup",
        "wall_ms",
        "detections",
        "num_slices",
        "num_attempts",
        "num_crop_predictions",
        "num_crop_batches",
        "num_roi_candidates",
        "num_roi_prefilter_dropped",
        *TIMING_KEYS,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(summary: dict) -> None:
    latency = summary["latency_ms"]
    workload = summary["workload"]
    print(
        "[speed] RL-SAHI cold E2E "
        f"mean={latency['mean']:.1f}ms median={latency['median']:.1f}ms "
        f"p90={latency['p90']:.1f}ms p95={latency['p95']:.1f}ms "
        f"speed={summary['images_per_second']:.2f}img/s"
    )
    print(
        "[speed] mean phases: "
        f"full={summary['timing_ms']['initial_detection_ms']['mean']:.1f}ms "
        f"rollout={summary['timing_ms']['rollout_ms']['mean']:.1f}ms "
        f"prefilter={summary['timing_ms']['roi_prefilter_ms']['mean']:.1f}ms "
        f"gate={summary['timing_ms']['candidate_evaluation_ms']['mean']:.1f}ms "
        f"crop={summary['timing_ms']['crop_inference_ms']['mean']:.1f}ms "
        f"merge={summary['timing_ms']['merge_ms']['mean']:.1f}ms"
    )
    print(
        "[speed] mean workload: "
        f"attempts={workload['attempts_mean']:.2f} "
        f"crop_predictions={workload['crop_predictions_mean']:.2f} "
        f"accepted_slices={workload['accepted_slices_mean']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure cold end-to-end RL-SAHI latency only. "
            "No baselines, AP metrics, detection cache, or prediction-file writes are used."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--image", type=Path, default=None, help="Measure one image instead of a split.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=None, help="Override infer.device, e.g. cuda or cuda:0.")
    parser.add_argument("--policy-device", default=None, help="Override infer.policy_device.")
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Images to measure (default: 100). Set -1 to measure the entire split.",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up images excluded from statistics.")
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive")
    if args.limit == 0:
        raise ValueError("--limit must be positive or -1 for the full split")

    project_cfg = load_default_config(args.config, ROOT)
    configured_device = project_cfg.optional_str("infer", "device")
    device = args.device or configured_device
    policy_device = args.policy_device or project_cfg.optional_str("infer", "policy_device") or device
    print_device_info("speed", device)
    print_device_info("speed-policy", policy_device)

    image_arg = None if args.image is None else _resolve_cli_path(args.image, ROOT)
    images = _select_images(project_cfg.path_value("image_root"), args.split, image_arg, args.limit)
    warmup = min(int(args.warmup), max(len(images) - 1, 0))
    checkpoint = (
        project_cfg.path_value("checkpoint")
        if args.checkpoint is None
        else _resolve_cli_path(args.checkpoint, ROOT)
    )
    output_dir = (
        ROOT / "runs" / "speed" / ("image" if image_arg is not None else args.split)
        if args.out_dir is None
        else _resolve_cli_path(args.out_dir, ROOT)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_cfg = _build_inference_config(project_cfg, device, policy_device)
    crop_weights = _optional_path(project_cfg, "crop_weights")
    full_weights = _optional_path(project_cfg, "full_weights")
    load_start = time.perf_counter()
    inferencer = AdaptiveSahiInferencer(
        weights=project_cfg.path_value("weights"),
        checkpoint=checkpoint,
        crop_weights=crop_weights,
        full_weights=full_weights,
        cfg=inference_cfg,
    )
    detector_device = resolve_torch_device(device)
    _sync_cuda(detector_device)
    model_load_ms = (time.perf_counter() - load_start) * 1000.0

    print(
        f"[speed] measuring RL-SAHI only: {len(images)} images, warmup={warmup}, "
        "cache=off, prediction/metadata/visualization writes=off"
    )
    records: list[dict] = []
    for index, image_path in enumerate(images, start=1):
        is_warmup = index <= warmup
        _sync_cuda(detector_device)
        start = time.perf_counter()
        meta = inferencer.infer_image(
            image_path=image_path,
            out_dir=output_dir,
            cache_root=None,
            split=None,
            use_cache=False,
        )
        _sync_cuda(detector_device)
        wall_ms = (time.perf_counter() - start) * 1000.0
        timing = meta.get("timing", {})
        record = {
            "index": index,
            "image": image_path.name,
            "is_warmup": is_warmup,
            "wall_ms": wall_ms,
            "detections": int(meta.get("detections", 0)),
            "num_slices": int(meta.get("num_slices", 0)),
            "num_attempts": int(meta.get("num_attempts", 0)),
            "num_crop_predictions": int(meta.get("num_crop_predictions", 0)),
            "num_crop_batches": int(meta.get("num_crop_batches", 0)),
            "num_roi_candidates": int(meta.get("num_roi_candidates", 0)),
            "num_roi_prefilter_dropped": int(meta.get("num_roi_prefilter_dropped", 0)),
            **{key: float(timing.get(key, 0.0)) for key in TIMING_KEYS},
        }
        records.append(record)
        if index == 1 or index % args.log_interval == 0 or index == len(images):
            status = "warmup" if is_warmup else "measure"
            print(
                f"[speed] {status} {index}/{len(images)} {image_path.name}: "
                f"wall={wall_ms:.1f}ms full={record['initial_detection_ms']:.1f}ms "
                f"crop={record['crop_inference_ms']:.1f}ms "
                f"attempts={record['num_attempts']} crops={record['num_crop_predictions']} "
                f"accepted={record['num_slices']}"
            )

    measured = [row for row in records if not row["is_warmup"]]
    latency = _stats(row["wall_ms"] for row in measured)
    timing_stats = {
        key: _stats(row[key] for row in measured)
        for key in TIMING_KEYS
    }
    summary = {
        "mode": "rl_sahi_only_cold_end_to_end",
        "notes": {
            "model_load_excluded_from_latency": True,
            "detection_cache_enabled": False,
            "prediction_metadata_visualization_writes_enabled": False,
            "full_prediction_reuses_initial_detection_when_weights_match": True,
        },
        "model_load_ms": model_load_ms,
        "images_total": len(images),
        "warmup_images": warmup,
        "measured_images": len(measured),
        "device": str(detector_device),
        "policy_device": str(resolve_torch_device(policy_device)),
        "inference_config": asdict(inference_cfg),
        "latency_ms": latency,
        "images_per_second": float(1000.0 / max(latency["mean"], 1e-9)),
        "timing_ms": timing_stats,
        "workload": {
            "attempts_mean": float(np.mean([row["num_attempts"] for row in measured])),
            "crop_predictions_mean": float(np.mean([row["num_crop_predictions"] for row in measured])),
            "accepted_slices_mean": float(np.mean([row["num_slices"] for row in measured])),
            "detections_mean": float(np.mean([row["detections"] for row in measured])),
        },
    }
    _write_csv(output_dir / "per_image.csv", records)
    (output_dir / "speed.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_summary(summary)
    print(f"[speed] wrote {output_dir / 'speed.json'} and {output_dir / 'per_image.csv'}")


if __name__ == "__main__":
    main()
