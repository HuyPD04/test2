from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "rl_no_prefilter",
    "rl_with_prefilter",
    "rl_with_prefilter_stop_gate",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RL-SAHI ablation benchmark.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/benchmark/ablation_val"),
    )
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir

    for variant in VARIANTS:
        command = [
            sys.executable,
            str(ROOT / "scripts" / "benchmark.py"),
            "--config",
            str(ROOT / "configs" / "ablations" / f"{variant}.yaml"),
            "--split",
            args.split,
            "--out-dir",
            str(out_dir / variant),
        ]
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        if args.no_cache:
            command.append("--no-cache")
        print(f"[ablation] running {variant}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    table_command = [
        sys.executable,
        str(ROOT / "scripts" / "build_ablation_table.py"),
        "--input-dir",
        str(out_dir),
        "--output",
        str(out_dir / "ablation_table.md"),
    ]
    subprocess.run(table_command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
