from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.cache import (
    detection_cache_is_current,
    detection_cache_metadata,
    detection_cache_path,
    hard_region_cache_is_current,
    hard_region_cache_metadata,
    hard_region_cache_path,
)
from rl_sahi.common.class_mapping import ClassMapping
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import iter_images
from rl_sahi.rl.state_config import StateConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_default_config(None, ROOT)
    state_cfg = cfg.dataclass_instance("state", StateConfig)
    detect_cfg = cfg.section("detect")
    hard_cfg = cfg.section("hard_region")
    class_mapping = ClassMapping.from_config(cfg.section("classes"))
    target_classes = cfg.target_classes()
    detection_metadata = detection_cache_metadata(
        weights=cfg.path_value("weights"),
        imgsz=int(detect_cfg["imgsz"]),
        conf=float(detect_cfg["conf"]),
        iou=float(detect_cfg["iou"]),
        max_det=int(detect_cfg["max_det"]),
        feature_layers=cfg.feature_layers("detect"),
        aux_grid_size=int(state_cfg.grid_size),
        spatial_feature_channels=int(state_cfg.spatial_feature_channels),
    )
    hard_metadata = hard_region_cache_metadata(
        detection_metadata=detection_metadata,
        small_area_ratio=float(hard_cfg["small_area_ratio"]),
        small_area_percentile=(
            None if hard_cfg.get("small_area_percentile") in (None, "")
            else float(hard_cfg["small_area_percentile"])
        ),
        match_iou=float(hard_cfg["match_iou"]),
        min_detect_score=float(hard_cfg["min_detect_score"]),
        target_classes=target_classes,
        class_mapping=class_mapping,
    )
    splits = (args.split,) if args.split else ("train", "val", "test")
    failed = False
    for split in splits:
        images = iter_images(cfg.path_value("image_root"), split=split, limit=args.limit)
        current_detection = 0
        current_hard = 0
        for image in images:
            current_detection += int(
                detection_cache_is_current(
                    detection_cache_path(cfg.path_value("cache_root"), split, image),
                    detection_metadata,
                )
            )
            current_hard += int(
                hard_region_cache_is_current(
                    hard_region_cache_path(cfg.path_value("cache_root"), split, image),
                    hard_metadata,
                )
            )
        print(
            f"[cache-check] {split}: detections={current_detection}/{len(images)} "
            f"hard_regions={current_hard}/{len(images)}"
        )
        failed = failed or current_detection != len(images) or current_hard != len(images)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
