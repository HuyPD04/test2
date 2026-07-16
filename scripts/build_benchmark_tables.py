from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAPER_TEST_ROWS = [
    ("ASAHI (4 slices)", 23.9, 38.7, 17.1, 5.19),
    ("ASAHI (6 slices)", 28.5, 41.6, 22.0, 4.98),
    ("ASAHI (12 slices)", 29.3, 41.9, 22.8, 2.98),
    ("ASAHI (15 slices)", 27.2, 40.9, 21.3, 2.39),
    ("ASAHI (adaptive)", 30.4, 45.6, 25.2, 4.88),
]

# One representative configuration per paper, all reported on VisDrone val.
PAPER_VAL_ROWS = [
    ("ClusDet (ResNeXt-101, multi-scale)", 32.4, 56.2, 31.6, "ICCV 2019"),
    ("AdaZoom+CT (Cascade R-CNN, ResNeXt-101)", 40.33, 66.94, 41.77, "arXiv:2106.10409"),
    ("QueryDet (RetinaNet-50, CSQ)", 28.32, 48.14, 28.75, "CVPR 2022"),
    ("AD-Det* (ResNeXt-101)", 37.5, 60.9, 39.2, "Remote Sensing 2025"),
    ("TPH+ASAHI", 36.0, 56.8, 28.2, "arXiv:2604.19233"),
]

CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(data: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not data:
        return {}
    return {str(row["method"]): row for row in data.get("results", [])}


def _pct(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}"


def _num(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _local_accuracy_row(label: str, row: dict[str, Any] | None) -> str:
    if row is None:
        return f"| {label} | local | - | - | - | - |"
    return (
        f"| {label} | local | {_pct(row.get('AP'))} | {_pct(row.get('AP50'))} | "
        f"{_pct(row.get('AP75'))} | {_num(row.get('images_per_second'))} |"
    )


def build(
    test_data: dict[str, Any] | None,
    val_data: dict[str, Any] | None,
    ablation_markdown: str | None = None,
) -> str:
    test = _rows(test_data)
    val = _rows(val_data)
    lines = [
        "# Thesis benchmark tables",
        "",
        "AP and mAP denote the same COCO-style mean Average Precision in these tables. Local "
        "values use all 10 VisDrone classes and are shown as percentages.",
        "",
        "## Internal comparison - VisDrone2019-DET test-dev",
        "",
        "| Method | Source | AP | AP50 | AP75 | Speed (img/s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, ap, ap50, ap75, speed in PAPER_TEST_ROWS:
        lines.append(f"| {name} | ASAHI paper | {ap:.1f} | {ap50:.1f} | {ap75:.1f} | {speed:.2f} |")
    lines.append(_local_accuracy_row("YOLO11s only", test.get("yolo_full")))
    lines.append(_local_accuracy_row("RL-SAHI (proposed)", test.get("rl_sahi")))

    lines.extend(
        [
            "",
            "> ASAHI paper values are reference-only: its TPH-YOLOv5 detector and hardware differ "
            "from the local YOLO11s pipeline, so the speed values are not controlled hardware comparisons.",
            "",
            "## SOTA comparison - VisDrone2019-DET val",
            "",
            "| Method | AP | AP50 | AP75 | Source |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, ap, ap50, ap75, source in PAPER_VAL_ROWS:
        lines.append(f"| {name} | {ap:.2f} | {ap50:.2f} | {ap75:.2f} | {source} |")
    proposed_val = val.get("rl_sahi")
    if proposed_val:
        lines.append(
            f"| RL-SAHI (proposed, YOLO11s) | {_pct(proposed_val.get('AP'))} | "
            f"{_pct(proposed_val.get('AP50'))} | {_pct(proposed_val.get('AP75'))} | local run |"
        )
    else:
        lines.append("| RL-SAHI (proposed, YOLO11s) | - | - | - | local run pending |")

    local_methods = [
        ("YOLO11s only", "yolo_full"),
        ("SAHI budget 4", "fixed_grid_budget_4"),
        ("SAHI budget 6", "fixed_grid_budget_6"),
        ("SAHI budget 12", "fixed_grid_budget_12"),
        ("SAHI budget 15", "fixed_grid_budget_15"),
        ("RL-SAHI", "rl_sahi"),
    ]
    lines.extend(
        [
            "",
            "## Internal detection diagnostics - local test run",
            "",
            "| Method | Precision@0.50 | Recall@0.50 | Recall-small@0.50 | FP/image |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, key in local_methods:
        row = test.get(key)
        if row:
            lines.append(
                f"| {label} | {_pct(row.get('precision'))} | {_pct(row.get('recall'))} | "
                f"{_pct(row.get('small_recall'))} | {_num(row.get('fp_per_image'))} |"
            )
        else:
            lines.append(f"| {label} | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Internal efficiency - local test run",
            "",
            "| Method | Latency (ms/image) | Speed (img/s) | Slices/image | Detector calls/image | Effective GFLOPs |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in local_methods:
        row = test.get(key)
        if row:
            lines.append(
                f"| {label} | {_num(row.get('end_to_end_ms_per_image'), 1)} | "
                f"{_num(row.get('images_per_second'))} | {_num(row.get('crops_per_image'))} | "
                f"{_num(row.get('detector_calls_per_image'))} | {_num(row.get('effective_gflops'), 1)} |"
            )
        else:
            lines.append(f"| {label} | - | - | - | - | - |")

    lines.extend(
        [
            "",
            "## Per-class AP - proposed method",
            "",
            "| Class | Test AP | Test AP50 | Val AP | Val AP50 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    proposed_test = test.get("rl_sahi", {})
    proposed_val = val.get("rl_sahi", {})
    for class_id, class_name in CLASS_NAMES.items():
        lines.append(
            f"| {class_id}: {class_name} | {_pct(proposed_test.get(f'AP_class_{class_id}'))} | "
            f"{_pct(proposed_test.get(f'AP50_class_{class_id}'))} | "
            f"{_pct(proposed_val.get(f'AP_class_{class_id}'))} | "
            f"{_pct(proposed_val.get(f'AP50_class_{class_id}'))} |"
        )

    if ablation_markdown:
        ablation_lines = ablation_markdown.strip().splitlines()
        if ablation_lines and ablation_lines[0].startswith("# "):
            ablation_lines[0] = "## " + ablation_lines[0][2:]
        lines.extend(["", *ablation_lines])

    lines.extend(
        [
            "",
            "## Protocol and provenance notes",
            "",
            "- Local test split: VisDrone2019-DET test-dev (1,610 labeled images).",
            "- Local val split: VisDrone2019-DET val (548 labeled images).",
            "- Local AP is the mean over IoU 0.50:0.95; AP50 and AP75 use their single IoU thresholds.",
            "- Effective GFLOPs follows Thesis.pdf: agent GFLOPs + detector GFLOPs x (1 + slices). "
            "YOLO11s is configured as 21.5 GFLOPs/pass; agent FLOPs are currently treated as negligible (0.0).",
            "- `2602.07512v2.pdf` is ZoomDet, not ASAHI. ASAHI values come from `2604.19233v1.pdf`.",
            "- Thesis.pdf retains all 10 classes, but its experimental-settings/results pages are accidentally "
            "replaced by MiniMedMind content; no result was copied from those corrupted pages.",
            "- The current checkpoint predates the latest class/provenance schema and README marks it as stale; "
            "local RL-SAHI numbers are reproducible preliminary results, not a substitute for the planned retrain.",
            "- `scripts/audit_splits.py` reports 24 shared sequence prefixes between train and val. The local val "
            "row must not be claimed as a final leak-free SOTA result until the split is repaired and the model is retrained.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build thesis-ready Markdown benchmark tables.")
    parser.add_argument("--test-json", type=Path)
    parser.add_argument("--val-json", type=Path)
    parser.add_argument("--ablation-table", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ablation_markdown = None
    if args.ablation_table is not None and args.ablation_table.exists():
        ablation_markdown = args.ablation_table.read_text(encoding="utf-8")
    args.output.write_text(
        build(_load(args.test_json), _load(args.val_json), ablation_markdown),
        encoding="utf-8",
    )
    print(f"[tables] wrote {args.output}")


if __name__ == "__main__":
    main()
