from __future__ import annotations

import argparse
from pathlib import Path
import sys


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))

from anchor_zoom_rl.config import load_config
from anchor_zoom_rl.runtime.data import iter_images
from anchor_zoom_rl.runtime.inferencer import AnchorZoomInferencer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run anchor-conditioned sequential sliced inference."
    )
    parser.add_argument(
        "--config", type=Path, default=METHOD_ROOT / "configs" / "default.yaml"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.detector.device = args.device
    if args.visualize:
        cfg.inference.save_visualization = True
    if args.no_save:
        cfg.inference.save_predictions = False
        cfg.inference.save_metadata = False
        cfg.inference.save_visualization = False
    checkpoint = _resolve_optional(args.checkpoint, METHOD_ROOT)
    output_dir = _resolve_optional(args.output_dir, METHOD_ROOT)
    inferencer = AnchorZoomInferencer(cfg, checkpoint=checkpoint)

    if args.image is not None:
        image_path = args.image.resolve()
        images = [image_path]
        split = "custom"
    else:
        split = str(args.split)
        images = iter_images(cfg.paths.image_root, split, args.limit)
    for image_path in images:
        result = inferencer.infer_image(
            image_path,
            split=split,
            output_dir=output_dir,
            save=not args.no_save,
        )
        print(
            f"[infer] {image_path.name}: boxes={len(result.detections)} "
            f"crops={len(result.accepted_rois)}/{len(result.attempted_rois)} "
            f"stop={result.stop_reason} "
            f"full={result.timing['full_detection_ms']:.1f}ms "
            f"crop={result.timing['crop_detection_ms']:.1f}ms "
            f"policy={result.timing['policy_ms']:.2f}ms "
            f"e2e={result.timing['end_to_end_ms']:.1f}ms"
        )


def _resolve_optional(path: Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    return path.resolve() if path.is_absolute() else (root / path).resolve()


if __name__ == "__main__":
    main()
