from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import iter_images
from rl_sahi.inference.visualize import save_inference_visual


def _load_prediction_txt(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boxes: list[list[float]] = []
    scores: list[float] = []
    classes: list[float] = []
    sources: list[int] = []
    if not path.exists():
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        classes.append(float(parts[0]))
        scores.append(float(parts[1]))
        boxes.append([float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])])
        sources.append(int(parts[6]))
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(scores, dtype=np.float32),
        np.asarray(classes, dtype=np.float32),
        np.asarray(sources, dtype=np.int32),
    )


def _rois_from_meta(meta: dict, accepted: bool) -> np.ndarray:
    rois = [
        row["roi"]
        for row in meta.get("slices", [])
        if bool(row.get("accepted", False)) is accepted and "roi" in row
    ]
    return np.asarray(rois, dtype=np.float32).reshape(-1, 4)


def _visualize_one(image_path: Path, infer_dir: Path, out_dir: Path | None) -> Path:
    meta_path = infer_dir / "metadata" / f"{image_path.stem}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prediction_file = meta.get("prediction_file")
    pred_path = Path(prediction_file) if prediction_file else infer_dir / "detections" / f"{image_path.stem}.txt"
    if not pred_path.is_absolute():
        pred_path = ROOT / pred_path
    boxes, _scores, _classes, sources = _load_prediction_txt(pred_path)
    accepted_rois = _rois_from_meta(meta, True)
    rejected_rois = _rois_from_meta(meta, False)
    output_root = out_dir if out_dir is not None else infer_dir / "visualizations"
    out_path = output_root / f"{image_path.stem}.jpg"
    save_inference_visual(image_path, boxes, sources, accepted_rois, rejected_rois, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render RL-SAHI inference visualizations from saved metadata.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--infer-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_default_config(args.config, ROOT)
    infer_dir = args.infer_dir if args.infer_dir is not None else cfg.path_value("infer_out_dir")
    if not infer_dir.is_absolute():
        infer_dir = ROOT / infer_dir
    out_dir = args.out_dir
    if out_dir is not None and not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    if args.image is not None:
        image_path = args.image if args.image.is_absolute() else ROOT / args.image
        images = [image_path]
    else:
        if args.split is None:
            raise ValueError("Use --image for one image or --split train/val/test for a dataset split.")
        images = iter_images(cfg.path_value("image_root"), split=args.split, limit=args.limit)

    for image_path in images:
        out_path = _visualize_one(image_path, infer_dir, out_dir)
        print(f"[visualize] {image_path.name}: {out_path}")


if __name__ == "__main__":
    main()
