from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.config import load_default_config
from rl_sahi.common.data import iter_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_default_config(args.config, ROOT)
    sequences = {
        split: {
            image.stem.split("_", 1)[0]
            for image in iter_images(cfg.path_value("image_root"), split=split)
        }
        for split in ("train", "val", "test")
    }
    failed = False
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(sequences[left] & sequences[right])
        print(f"[split-audit] {left}/{right}: {len(overlap)} shared sequences")
        if overlap:
            print("  " + ", ".join(overlap))
            failed = True
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
