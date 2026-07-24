from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))

from anchor_zoom_rl.config import load_config
from anchor_zoom_rl.runtime.data import (
    iter_images,
    label_path_for,
    load_yolo_labels,
    read_image,
)
from anchor_zoom_rl.runtime.inferencer import AnchorZoomInferencer
from anchor_zoom_rl.runtime.metrics import AP50Accumulator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate AP50 and latency for the anchor-zoom policy."
    )
    parser.add_argument(
        "--config", type=Path, default=METHOD_ROOT / "configs" / "default.yaml"
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.detector.device = args.device
    cfg.inference.save_predictions = False
    cfg.inference.save_metadata = False
    cfg.inference.save_visualization = False
    checkpoint = (
        None
        if args.checkpoint is None
        else (
            args.checkpoint.resolve()
            if args.checkpoint.is_absolute()
            else (METHOD_ROOT / args.checkpoint).resolve()
        )
    )
    inferencer = AnchorZoomInferencer(cfg, checkpoint=checkpoint)
    images = iter_images(cfg.paths.image_root, args.split, args.limit)
    accumulator = AP50Accumulator(cfg.detector.target_classes, cfg.reward.match_iou)
    timings: list[dict[str, float]] = []
    crop_counts: list[int] = []
    for index, image_path in enumerate(images, start=1):
        result = inferencer.infer_image(image_path, split=args.split, save=False)
        image_shape = read_image(image_path).shape[:2]
        ground_truth = load_yolo_labels(
            label_path_for(image_path, cfg.paths.label_root, args.split),
            image_shape,
        ).filter_classes(cfg.detector.target_classes)
        accumulator.update(result.detections, ground_truth)
        timings.append(result.timing)
        crop_counts.append(len(result.attempted_rois))
        print(
            f"[evaluate] {index}/{len(images)} {image_path.name} "
            f"crops={crop_counts[-1]} e2e={result.timing['end_to_end_ms']:.1f}ms"
        )

    summary = accumulator.compute()
    summary.update(
        {
            "images": len(images),
            "mean_crops": _mean(crop_counts),
            "latency_ms_per_image": _mean(
                [item["method_latency_ms"] for item in timings]
            ),
            "initial_state_ms_per_image": _mean(
                [item["initial_state_ms"] for item in timings]
            ),
            "end_to_end_ms_per_image": _mean(
                [item["end_to_end_ms"] for item in timings]
            ),
        }
    )
    output = (
        args.output.resolve()
        if args.output is not None and args.output.is_absolute()
        else (
            (METHOD_ROOT / args.output).resolve()
            if args.output is not None
            else cfg.paths.output_dir / f"evaluation_{args.split}.json"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[evaluate] output={output}")


def _mean(values) -> float:
    return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    main()
