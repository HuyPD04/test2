from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    checkpoint: Path
    spatial: bool
    detection: bool
    history: bool
    outcome: bool
    cost_overlap: bool
    action_mask: bool


VARIANTS = (
    Variant(
        "full",
        "Full RL-SAHI",
        Path("runs/dqn/best.pt"),
        True,
        True,
        True,
        True,
        True,
        True,
    ),
    Variant(
        "no_spatial",
        "w/o spatial feature",
        Path("runs/dqn_ablation/no_spatial/best.pt"),
        False,
        True,
        True,
        True,
        True,
        True,
    ),
    Variant(
        "no_detection_map",
        "w/o detection map",
        Path("runs/dqn_ablation/no_detection_map/best.pt"),
        True,
        False,
        True,
        True,
        True,
        True,
    ),
    Variant(
        "no_history",
        "w/o history",
        Path("runs/dqn_ablation/no_history/best.pt"),
        True,
        True,
        False,
        True,
        True,
        True,
    ),
    Variant(
        "no_outcome_reward",
        "w/o outcome reward",
        Path("runs/dqn_ablation/no_outcome_reward/best.pt"),
        True,
        True,
        True,
        False,
        True,
        True,
    ),
    Variant(
        "no_cost_overlap",
        "w/o cost/overlap",
        Path("runs/dqn_ablation/no_cost_overlap/best.pt"),
        True,
        True,
        True,
        True,
        False,
        True,
    ),
    Variant(
        "no_action_mask",
        "w/o action mask",
        Path("runs/dqn_ablation/no_action_mask/best.pt"),
        True,
        True,
        True,
        True,
        True,
        False,
    ),
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_result(path: Path, method: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return next(
        (row for row in payload.get("results", []) if str(row.get("method")) == method),
        None,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("manifest", {})


def _pct(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}"


def _delta(value: Any, reference: Any) -> str:
    if value is None or reference is None:
        return "-"
    return f"{100.0 * (float(value) - float(reference)):+.2f}"


def _num(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _yn(value: bool) -> str:
    return "Yes" if value else "No"


def build_table(input_dir: Path) -> str:
    full = _load_result(input_dir / "full" / "benchmark.json", "rl_sahi")
    yolo = _load_result(input_dir / "full" / "benchmark.json", "yolo_full")
    manifests = [
        _load_manifest(input_dir / variant.key / "benchmark.json")
        for variant in VARIANTS
        if (input_dir / variant.key / "benchmark.json").exists()
    ]
    devices = sorted(
        {
            str(manifest.get("device"))
            for manifest in manifests
            if manifest.get("device") not in (None, "")
        }
    )
    checkpoints = sorted(
        {
            Path(str(manifest.get("checkpoint", {}).get("path", ""))).name
            for manifest in manifests
            if isinstance(manifest.get("checkpoint"), dict)
        }
    )
    device_note = ", ".join(devices) if devices else "pending"
    checkpoint_note = ", ".join(checkpoints) if checkpoints else "pending"

    lines = [
        "# RL-SAHI component ablation - VisDrone2019-DET val",
        "",
        "Each row uses the same detector, validation split, inference thresholds, crop "
        "acceptance settings, and evaluator. Only the DQN checkpoint changes.",
        "",
        f"Device: `{device_note}`. Checkpoint file name(s): `{checkpoint_note}`.",
    ]
    if yolo is not None:
        lines.append(
            f"YOLO full-image reference: AP={_pct(yolo.get('AP'))}, "
            f"AP50={_pct(yolo.get('AP50'))}, "
            f"Recall-small@0.50={_pct(yolo.get('small_recall'))}."
        )
    lines.extend(
        [
            "",
            "| Variant | Spatial feature | Detection map | History | Outcome reward | Cost/overlap | Action mask | AP | Delta AP | AP50 | AP75 | Recall-small@0.50 | FP/image | Crops/image | Speed (img/s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    reference_ap = None if full is None else full.get("AP")
    for variant in VARIANTS:
        row = _load_result(input_dir / variant.key / "benchmark.json", "rl_sahi")
        if row is None:
            lines.append(
                f"| {variant.label} | {_yn(variant.spatial)} | {_yn(variant.detection)} | "
                f"{_yn(variant.history)} | {_yn(variant.outcome)} | "
                f"{_yn(variant.cost_overlap)} | {_yn(variant.action_mask)} | - | - | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {variant.label} | {_yn(variant.spatial)} | {_yn(variant.detection)} | "
            f"{_yn(variant.history)} | {_yn(variant.outcome)} | "
            f"{_yn(variant.cost_overlap)} | {_yn(variant.action_mask)} | "
            f"{_pct(row.get('AP'))} | {_delta(row.get('AP'), reference_ap)} | "
            f"{_pct(row.get('AP50'))} | {_pct(row.get('AP75'))} | "
            f"{_pct(row.get('small_recall'))} | {_num(row.get('fp_per_image'))} | "
            f"{_num(row.get('crops_per_image'))} | {_num(row.get('images_per_second'))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark trained RL-SAHI component ablations.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark_component_ablation.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/benchmark/component_ablation_val"))
    parser.add_argument("--policy-device", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Only rebuild the table from existing outputs.")
    args = parser.parse_args()

    out_dir = _resolve(args.out_dir)
    config = _resolve(args.config)

    if not args.skip_run:
        for variant in VARIANTS:
            checkpoint = _resolve(variant.checkpoint)
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint for {variant.key}: {checkpoint}")
            command = [
                sys.executable,
                str(ROOT / "scripts" / "benchmark.py"),
                "--config",
                str(config),
                "--split",
                args.split,
                "--checkpoint",
                str(checkpoint),
                "--out-dir",
                str(out_dir / variant.key),
            ]
            if args.limit is not None:
                command.extend(("--limit", str(args.limit)))
            if args.policy_device is not None:
                command.extend(("--policy-device", str(args.policy_device)))
            if args.no_cache:
                command.append("--no-cache")
            print(f"[component-ablation] running {variant.key}", flush=True)
            subprocess.run(command, cwd=ROOT, check=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "component_ablation_table.md"
    table_path.write_text(build_table(out_dir), encoding="utf-8")
    print(f"[component-ablation] wrote {table_path}")


if __name__ == "__main__":
    main()
