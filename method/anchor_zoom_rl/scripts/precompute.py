from __future__ import annotations

import argparse
from pathlib import Path
import sys


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))

from anchor_zoom_rl.config import load_config
from anchor_zoom_rl.runtime.data import (
    iter_images,
    label_path_for,
    load_yolo_labels,
    read_image,
)
from anchor_zoom_rl.runtime.prediction import DetectionRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute full-image detections and hard-region caches."
    )
    parser.add_argument(
        "--config", type=Path, default=METHOD_ROOT / "configs" / "default.yaml"
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.detector.device = args.device
    runner = DetectionRunner(cfg)
    images = iter_images(cfg.paths.image_root, args.split, args.limit)
    print(f"[precompute] detections={cfg.paths.cache_dir / 'detections' / args.split}")
    print(f"[precompute] hard_regions={cfg.paths.cache_dir / 'hard_regions' / args.split}")
    for index, image_path in enumerate(images, start=1):
        image = read_image(image_path)
        image_shape = image.shape[:2]
        label_path = label_path_for(image_path, cfg.paths.label_root, args.split)
        ground_truth = load_yolo_labels(label_path, image_shape).filter_classes(
            cfg.detector.target_classes
        )
        detections, elapsed_ms, detection_hit = runner.full(
            image, image_path, args.split, use_cache=True
        )
        regions, hard_hit = runner.hard_regions(
            detections,
            ground_truth,
            image_path,
            label_path,
            args.split,
            use_cache=True,
        )
        print(
            f"[precompute] {index}/{len(images)} {image_path.name}: "
            f"detections={len(detections)} hard={int(regions.hard_mask.sum())}/"
            f"{len(ground_truth)} time={elapsed_ms:.1f}ms "
            f"cache_detection={detection_hit} cache_hard={hard_hit}"
        )


if __name__ == "__main__":
    main()
