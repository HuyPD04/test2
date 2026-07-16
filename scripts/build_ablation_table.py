from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    ("rl_no_prefilter", "RL slicing", "No", "No"),
    ("rl_with_prefilter", "RL slicing + ROI pre-filter", "Yes", "No"),
    (
        "rl_with_prefilter_stop_gate",
        "RL slicing + ROI pre-filter + STOP gate",
        "Yes",
        "Yes",
    ),
)


def _load_result(path: Path, method: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return next(
        (row for row in payload.get("results", []) if str(row.get("method")) == method),
        None,
    )


def _load_device(path: Path) -> str | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("manifest", {}).get("device")
    return None if value in (None, "") else str(value)


def _pct(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}"


def _num(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def build(input_dir: Path) -> str:
    rows: list[tuple[str, str, str, dict[str, Any] | None]] = []
    yolo: dict[str, Any] | None = None
    devices: set[str] = set()
    for key, label, prefilter, stop_gate in VARIANTS:
        result_path = input_dir / key / "benchmark.json"
        device = _load_device(result_path)
        if device is not None:
            devices.add(device)
        yolo = yolo or _load_result(result_path, "yolo_full")
        rows.append((label, prefilter, stop_gate, _load_result(result_path, "rl_sahi")))

    device_note = ", ".join(sorted(devices)) if devices else "pending"
    lines = [
        "# RL-SAHI ablation - VisDrone2019-DET val",
        "",
        "All variants use the same detector, DQN checkpoint, validation images, confidence "
        "thresholds, merge settings, and evaluator. The STOP gate is "
        "`require_stop_for_acceptance`; regular crop utility checks remain enabled in every "
        "RL variant.",
        "",
        f"Execution device for the reported speed values: `{device_note}`. Accuracy metrics "
        "are comparable across these rows; do not compare this speed column with T4 paper results.",
        "",
        "| Configuration | ROI pre-filter | STOP gate | AP | AP50 | AP75 | Recall-small@0.50 | FP/image | Crops/image | Speed (img/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if yolo is not None:
        lines.append(
            f"| YOLO11s full-image | No | No | {_pct(yolo.get('AP'))} | "
            f"{_pct(yolo.get('AP50'))} | {_pct(yolo.get('AP75'))} | "
            f"{_pct(yolo.get('small_recall'))} | {_num(yolo.get('fp_per_image'))} | "
            f"0.00 | {_num(yolo.get('images_per_second'))} |"
        )
    else:
        lines.append("| YOLO11s full-image | No | No | - | - | - | - | - | 0.00 | - |")

    for label, prefilter, stop_gate, row in rows:
        if row is None:
            lines.append(
                f"| {label} | {prefilter} | {stop_gate} | - | - | - | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {label} | {prefilter} | {stop_gate} | {_pct(row.get('AP'))} | "
            f"{_pct(row.get('AP50'))} | {_pct(row.get('AP75'))} | "
            f"{_pct(row.get('small_recall'))} | {_num(row.get('fp_per_image'))} | "
            f"{_num(row.get('crops_per_image'))} | {_num(row.get('images_per_second'))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RL-SAHI validation ablation table.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("runs/benchmark/ablation_val"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/benchmark/ablation_val/ablation_table.md"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args.input_dir), encoding="utf-8")
    print(f"[ablation-table] wrote {args.output}")


if __name__ == "__main__":
    main()
