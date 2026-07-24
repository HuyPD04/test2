from __future__ import annotations

import argparse
from pathlib import Path
import sys


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))

from anchor_zoom_rl.config import load_config
from anchor_zoom_rl.runtime.trainer import AnchorZoomTrainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the independent anchor-conditioned sequential DQN."
    )
    parser.add_argument(
        "--config", type=Path, default=METHOD_ROOT / "configs" / "default.yaml"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Independent output directory for this training run.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.detector.device = args.device
    if args.out_dir is not None:
        out_dir = (
            args.out_dir.resolve()
            if args.out_dir.is_absolute()
            else (METHOD_ROOT / args.out_dir).resolve()
        )
        cfg.paths.output_dir = out_dir
        cfg.paths.checkpoint = out_dir / "checkpoints" / "best.pt"
    if args.resume is not None:
        cfg.train.resume = bool(args.resume)
    trainer = AnchorZoomTrainer(
        cfg,
        train_split=args.split,
        val_split=args.val_split,
        limit=args.limit,
    )
    checkpoint = trainer.train(args.episodes)
    print(f"[train] checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
