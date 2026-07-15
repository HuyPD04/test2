from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.device import print_device_info
from rl_sahi.eval.benchmark import BenchmarkConfig, benchmark_split
from rl_sahi.inference.config import InferenceConfig


def _int_tuple(value) -> tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


def _bool_value(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YOLO full, fixed-grid SAHI, and RL-SAHI.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--sampling", choices=("stratified", "sequential"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    infer_cfg = cfg.section("infer")
    device = cfg.optional_str("infer", "device")
    print_device_info("benchmark", device)
    benchmark_cfg = cfg.section("benchmark")
    target_classes = cfg.target_classes()
    class_mapping = ClassMapping.from_config(cfg.section("classes"))
    checkpoint = cfg.path_value("checkpoint") if args.checkpoint is None else args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    out_dir = args.out_dir if args.out_dir is not None else ROOT / "runs" / "benchmark" / args.split
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    rows = benchmark_split(
        weights=cfg.path_value("weights"),
        checkpoint=checkpoint,
        image_root=cfg.path_value("image_root"),
        label_root=cfg.path_value("label_root"),
        cache_root=cfg.path_value("cache_root"),
        split=args.split,
        infer_cfg=InferenceConfig(
            full_imgsz=int(infer_cfg["full_imgsz"]),
            slice_imgsz=int(infer_cfg["slice_imgsz"]),
            full_conf=float(infer_cfg["full_conf"]),
            output_conf=float(infer_cfg["output_conf"]),
            iou=float(infer_cfg["iou"]),
            merge_iou=float(infer_cfg["merge_iou"]),
            max_det=int(infer_cfg["max_det"]),
            device=device,
            feature_layers=cfg.feature_layers("infer"),
            min_slice_detections=int(infer_cfg.get("min_slice_detections", 1)),
            min_slice_utility=float(infer_cfg.get("min_slice_utility", 0.5)),
            min_new_detection_score=float(infer_cfg.get("min_new_detection_score", 0.45)),
            duplicate_iou=float(infer_cfg.get("duplicate_iou", infer_cfg.get("merge_iou", 0.5))),
            max_slice_attempts=int(infer_cfg.get("max_slice_attempts", 0)),
            crop_batch_size=int(infer_cfg.get("crop_batch_size", 1)),
            max_consecutive_rejections=int(infer_cfg.get("max_consecutive_rejections", 0)),
            target_classes=target_classes,
            require_stop_for_acceptance=_bool_value(infer_cfg.get("require_stop_for_acceptance", True)),
            save_predictions=False,
            save_metadata=False,
            save_visualization=False,
            batched_inference=_bool_value(infer_cfg.get("batched_inference", False)),
            class_mapping=class_mapping,
        ),
        bench_cfg=BenchmarkConfig(
            iou_threshold=float(benchmark_cfg.get("iou_threshold", 0.5)),
            fixed_slice_fraction=float(benchmark_cfg.get("fixed_slice_fraction", 0.35)),
            fixed_overlap=float(benchmark_cfg.get("fixed_overlap", 0.2)),
            budgeted_crop_counts=_int_tuple(benchmark_cfg.get("budgeted_crop_counts", (4, 8, 12))),
            include_fixed_grid_full=_bool_value(benchmark_cfg.get("include_fixed_grid_full", True)),
            include_gated_variants=_bool_value(benchmark_cfg.get("include_gated_variants", True)),
            proposal_crop_count=int(benchmark_cfg.get("proposal_crop_count", 8)),
            proposal_min_conf=float(benchmark_cfg.get("proposal_min_conf", 0.01)),
            proposal_max_conf=float(benchmark_cfg.get("proposal_max_conf", 0.5)),
            proposal_peak_conf=float(benchmark_cfg.get("proposal_peak_conf", 0.25)),
            proposal_overlap_threshold=float(benchmark_cfg.get("proposal_overlap_threshold", 0.7)),
            small_area_ratio=(
                None if benchmark_cfg.get("small_area_ratio") in (None, "")
                else float(benchmark_cfg["small_area_ratio"])
            ),
            small_area_percentile=(
                None if benchmark_cfg.get("small_area_percentile") in (None, "")
                else float(benchmark_cfg["small_area_percentile"])
            ),
            sampling=str(args.sampling or benchmark_cfg.get("sampling", "stratified")),
            seed=int(args.seed if args.seed is not None else benchmark_cfg.get("seed", 42)),
            warmup_images=int(benchmark_cfg.get("warmup_images", 10)),
            detector_gflops=float(benchmark_cfg.get("detector_gflops", 21.5)),
            agent_gflops=float(benchmark_cfg.get("agent_gflops", 0.0)),
            target_classes=target_classes,
            class_mapping=class_mapping,
        ),
        out_dir=out_dir,
        limit=args.limit,
        use_cache=_bool_value(infer_cfg.get("use_cache", True)) and not args.no_cache,
    )
    for row in rows:
        print(
            f"[benchmark] {row['method']}: AP={row['AP']:.4f} "
            f"AP50={row['AP50']:.4f} AP75={row['AP75']:.4f} "
            f"small_recall={row['small_recall']:.4f} fp/image={row['fp_per_image']:.2f} "
            f"crops/image={row['crops_per_image']:.2f} "
            f"accepted={row['accepted_crops_per_image']:.2f} "
            f"latency={row['latency_ms_per_image']:.1f}ms "
            f"e2e={row['end_to_end_ms_per_image']:.1f}ms "
            f"speed={row['images_per_second']:.2f}img/s"
        )
    print(f"[benchmark] wrote {out_dir / 'benchmark.csv'}")


if __name__ == "__main__":
    main()
