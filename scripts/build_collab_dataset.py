from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.collab.dataset import CollaborativeDatasetConfig, build_yolo_crop_dataset
from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import iter_images


def _int_tuple(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip() == "":
        return None
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a YOLO-format crop dataset from RL-SAHI inference metadata."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--out-split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--metadata-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("data") / "collab_yolo")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--roi-source", default="accepted", choices=["accepted", "rejected", "all"])
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--min-box-size", type=float, default=2.0)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--target-classes", default=None, help="Optional comma-separated source label ids to keep.")
    parser.add_argument("--image-ext", default=".jpg")
    args = parser.parse_args()

    project_cfg = load_default_config(args.config, ROOT)
    infer_out_dir = project_cfg.path_value("infer_out_dir")
    metadata_dir = args.metadata_dir if args.metadata_dir is not None else infer_out_dir / "metadata"
    if not metadata_dir.is_absolute():
        metadata_dir = ROOT / metadata_dir

    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_split = args.out_split or args.split
    image_paths = iter_images(project_cfg.path_value("image_root"), split=args.split, limit=args.limit)

    summary = build_yolo_crop_dataset(
        metadata_dir=metadata_dir,
        image_root=project_cfg.path_value("image_root"),
        label_root=project_cfg.path_value("label_root"),
        out_dir=out_dir,
        split=out_split,
        image_paths=image_paths,
        cfg=CollaborativeDatasetConfig(
            roi_source=args.roi_source,
            min_visibility=args.min_visibility,
            min_box_size=args.min_box_size,
            include_empty=args.include_empty,
            target_classes=_int_tuple(args.target_classes),
            image_ext=args.image_ext,
        ),
    )

    print(f"[collab_dataset] wrote YOLO dataset: {out_dir}")
    print(f"[collab_dataset] split: {out_split}")
    for key, value in summary.as_dict().items():
        print(f"[collab_dataset] {key}: {value}")
    print(f"[collab_dataset] yaml: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()

