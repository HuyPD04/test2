from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    COCO = None
    COCOeval = None


CATEGORIES = [
    {"id": 1, "name": "pedestrian", "supercategory": "person"},
    {"id": 2, "name": "people", "supercategory": "person"},
    {"id": 3, "name": "bicycle", "supercategory": "vehicle"},
    {"id": 4, "name": "car", "supercategory": "vehicle"},
    {"id": 5, "name": "van", "supercategory": "vehicle"},
    {"id": 6, "name": "truck", "supercategory": "vehicle"},
    {"id": 7, "name": "tricycle", "supercategory": "vehicle"},
    {"id": 8, "name": "awning-tricycle", "supercategory": "vehicle"},
    {"id": 9, "name": "bus", "supercategory": "vehicle"},
    {"id": 10, "name": "motor", "supercategory": "vehicle"},
]


def get_image_shape(image_path: Path) -> tuple[int, int]:
    """Return (height, width) of an image file."""
    if not image_path.exists():
        return 1080, 1920  # Default fallback for VisDrone if image is missing
    if Image is not None:
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                return int(h), int(w)
        except Exception:
            pass
    if cv2 is not None:
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                return int(h), int(w)
        except Exception:
            pass
    return 1080, 1920


def convert_gt_to_coco(
    annotations_dir: Path,
    images_dir: Path | None,
    output_json: Path,
) -> tuple[dict[str, Any], dict[str, int]]:
    """
    Convert VisDrone ground truth .txt annotations to COCO JSON format.
    Returns (coco_dict, stem_to_image_id_mapping).
    """
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    txt_files = sorted(annotations_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {annotations_dir}")

    print(f"[GT Convert] Found {len(txt_files)} annotation files in {annotations_dir}")

    images_list: list[dict[str, Any]] = []
    annotations_list: list[dict[str, Any]] = []
    stem_to_id: dict[str, int] = {}
    ann_id = 1

    for idx, txt_path in enumerate(txt_files, start=1):
        stem = txt_path.stem
        stem_to_id[stem] = idx

        # Locate corresponding image for width/height
        img_h, img_w = 1080, 1920
        img_name = f"{stem}.jpg"
        if images_dir and images_dir.exists():
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.exists():
                    img_h, img_w = get_image_shape(candidate)
                    img_name = candidate.name
                    break

        images_list.append({
            "id": idx,
            "file_name": img_name,
            "width": img_w,
            "height": img_h,
        })

        # Read annotation lines
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
                if len(parts) < 6:
                    continue
                try:
                    x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                    score = float(parts[4])
                    category = int(float(parts[5]))
                except ValueError:
                    continue

                # In VisDrone: score=0 is ignore, category=0 is ignore region, category=11 is others
                if score <= 0.0 or category < 1 or category > 10:
                    continue
                if w <= 0.0 or h <= 0.0:
                    continue

                bbox = [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]
                area = round(w * h, 2)
                annotations_list.append({
                    "id": ann_id,
                    "image_id": idx,
                    "category_id": category,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "ignore": 0,
                })
                ann_id += 1

    coco_dict = {
        "images": images_list,
        "annotations": annotations_list,
        "categories": CATEGORIES,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2)

    print(f"[GT Convert] Saved {len(images_list)} images and {len(annotations_list)} annotations -> {output_json}")
    return coco_dict, stem_to_id


def convert_preds_to_coco(
    detections_dir: Path,
    stem_to_id: dict[str, int],
    output_json: Path,
) -> list[dict[str, Any]]:
    """
    Convert detection prediction .txt files to COCO JSON list format.
    Supports both space-separated YOLO format (class_id score x1 y1 x2 y2 source)
    and comma-separated official VisDrone format (x1, y1, w, h, score, class_id, -1, -1).
    """
    if not detections_dir.exists():
        raise FileNotFoundError(f"Detections directory not found: {detections_dir}")

    txt_files = sorted(detections_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {detections_dir}")

    print(f"[DT Convert] Found {len(txt_files)} detection files in {detections_dir}")

    predictions_list: list[dict[str, Any]] = []
    skipped_files = 0
    valid_boxes = 0

    for txt_path in txt_files:
        stem = txt_path.stem
        image_id = stem_to_id.get(stem)
        if image_id is None:
            skipped_files += 1
            continue

        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.replace(",", " ").split() if p.strip()]
                if len(parts) < 5:
                    continue
                try:
                    # Auto-detect format based on field positions
                    if float(parts[1]) <= 1.0 and float(parts[0]) <= 20 and float(parts[2]) > 1.0 and len(parts) >= 6:
                        # Space-separated YOLO format: class_id score x1 y1 x2 y2 [source]
                        class_id = int(float(parts[0]))
                        score = float(parts[1])
                        x1, y1, x2, y2 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                        category_id = class_id + 1  # Map 0..9 (YOLO) -> 1..10 (COCO/VisDrone)
                    else:
                        # Comma-separated official format: x1, y1, w, h, score, class_id, -1, -1
                        x1, y1, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                        score = float(parts[4])
                        category_id = int(float(parts[5]))  # Already 1..10
                except (ValueError, IndexError):
                    continue

                if w <= 0.0 or h <= 0.0 or score <= 0.0:
                    continue
                if category_id < 1 or category_id > 10:
                    continue

                bbox = [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)]
                predictions_list.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": round(score, 6),
                })
                valid_boxes += 1

    if skipped_files > 0:
        print(f"[DT Convert] Warning: Skipped {skipped_files} files not present in Ground Truth annotations.")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(predictions_list, f, indent=2)

    print(f"[DT Convert] Saved {valid_boxes} detection boxes across {len(txt_files) - skipped_files} images -> {output_json}")
    return predictions_list


def evaluate_coco(
    gt_json_path: Path,
    dt_json_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Run pycocotools COCOeval and output comprehensive metrics report."""
    if COCO is None or COCOeval is None:
        raise ImportError(
            "pycocotools is not installed in the current Python environment.\n"
            "Please install it (e.g., `pip install pycocotools` or use your conda environment `doan`)."
        )

    print(f"\n[COCO Eval] Loading Ground Truth from {gt_json_path}...")
    coco_gt = COCO(str(gt_json_path))

    print(f"[COCO Eval] Loading Predictions from {dt_json_path}...")
    coco_dt = coco_gt.loadRes(str(dt_json_path))

    print("[COCO Eval] Running evaluation (iouType='bbox')...")
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extract 12 standard overall metrics
    stats = coco_eval.stats
    overall_metrics = {
        "AP_0.50_0.95": round(float(stats[0]), 4),
        "AP_50": round(float(stats[1]), 4),
        "AP_75": round(float(stats[2]), 4),
        "AP_small": round(float(stats[3]), 4),
        "AP_medium": round(float(stats[4]), 4),
        "AP_large": round(float(stats[5]), 4),
        "AR_max1": round(float(stats[6]), 4),
        "AR_max10": round(float(stats[7]), 4),
        "AR_max100": round(float(stats[8]), 4),
        "AR_small": round(float(stats[9]), 4),
        "AR_medium": round(float(stats[10]), 4),
        "AR_large": round(float(stats[11]), 4),
    }

    # Extract class-wise AP and AP50
    precision = coco_eval.eval["precision"]  # shape: [T, R, K, A, M]
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    class_metrics = {}

    table_lines = [
        "\n" + "=" * 65,
        f"{'Category':<18} | {'ID':<4} | {'AP50':<12} | {'mAP (0.50:0.95)':<15}",
        "-" * 65,
    ]

    for k_idx, cat in enumerate(cats):
        # mAP (0.50:0.95) across all areas (A=0), max det (M=-1)
        s_map = precision[:, :, k_idx, 0, -1]
        s_map = s_map[s_map > -1]
        cat_map = float(np.mean(s_map)) if len(s_map) > 0 else 0.0

        # AP50 (T=0 is IoU=0.50)
        s_50 = precision[0, :, k_idx, 0, -1]
        s_50 = s_50[s_50 > -1]
        cat_ap50 = float(np.mean(s_50)) if len(s_50) > 0 else 0.0

        class_metrics[cat["name"]] = {
            "id": cat["id"],
            "AP50": round(cat_ap50, 4),
            "mAP_0.50_0.95": round(cat_map, 4),
        }
        table_lines.append(f"{cat['name']:<18} | {cat['id']:<4} | {cat_ap50:<12.4f} | {cat_map:<15.4f}")

    table_lines.append("=" * 65)
    summary_text = "\n".join(table_lines)
    print(summary_text)

    # Save comprehensive JSON report
    report = {
        "overall": overall_metrics,
        "per_class": class_metrics,
        "files": {
            "ground_truth_json": str(gt_json_path),
            "predictions_json": str(dt_json_path),
        },
    }
    report_json_path = out_dir / "eval_results.json"
    with report_json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_txt_path = out_dir / "eval_summary.txt"
    report_txt_path.write_text(summary_text.lstrip() + "\n", encoding="utf-8")

    print(f"\n[COCO Eval] Complete! Results saved to:\n  - {report_json_path}\n  - {report_txt_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VisDrone GT and YOLO detections to COCO JSON format and run COCOeval."
    )
    root_dir = Path(__file__).resolve().parents[1]

    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=root_dir / "data" / "raw" / "VisDrone2019-DET-test-dev" / "annotations",
        help="Directory containing ground truth .txt files.",
    )
    parser.add_argument(
        "--img-dir",
        type=Path,
        default=root_dir / "data" / "raw" / "VisDrone2019-DET-test-dev" / "images",
        help="Directory containing image files (to read dimensions).",
    )
    parser.add_argument(
        "--dt-dir",
        type=Path,
        default=root_dir / "runs" / "infer" / "detections",
        help="Directory containing detection prediction .txt files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root_dir / "benchmark_coco",
        help="Output directory to store JSONs and evaluation reports.",
    )
    parser.add_argument(
        "--gt-json-name",
        type=str,
        default="visdrone_test_dev_gt.json",
        help="Filename for generated COCO GT JSON.",
    )
    parser.add_argument(
        "--dt-json-name",
        type=str,
        default="predictions_coco.json",
        help="Filename for generated COCO predictions JSON.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Only convert to JSON without running COCOeval.",
    )

    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_json = out_dir / args.gt_json_name
    dt_json = out_dir / args.dt_json_name

    print("=" * 65)
    print(" VISDRONE -> COCO BENCHMARK PIPELINE")
    print("=" * 65)

    # 1. Convert Ground Truth
    _, stem_to_id = convert_gt_to_coco(args.gt_dir, args.img_dir, gt_json)

    # 2. Convert Predictions
    convert_preds_to_coco(args.dt_dir, stem_to_id, dt_json)

    # 3. Evaluate
    if not args.skip_eval:
        evaluate_coco(gt_json, dt_json, out_dir)


if __name__ == "__main__":
    main()
