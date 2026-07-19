from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.config import load_default_config


DEFAULT_IMAGE = ROOT / r"data\raw\images\test\0000074_06746_d_0000013.jpg"
VISDRONE_CLASSES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


def image_to_label_path(image_path: Path, image_root: Path, label_root: Path) -> Path:
    """Map data/raw/images/<split>/x.jpg to data/raw/labels/<split>/x.txt."""
    try:
        relative_path = image_path.relative_to(image_root)
    except ValueError as exc:
        raise ValueError(
            f"Image {image_path} is outside configured image root {image_root}; "
            "provide --label explicitly."
        ) from exc
    return (label_root / relative_path).with_suffix(".txt")


def read_yolo_labels(
    label_path: Path, image_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Read class/xywhn rows and return classes plus pixel xyxy boxes."""
    rows: list[list[float]] = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(
                f"Invalid YOLO label at {label_path}:{line_number}: expected 5 values"
            )
        try:
            rows.append([float(value) for value in parts[:5]])
        except ValueError as exc:
            raise ValueError(
                f"Invalid number at {label_path}:{line_number}: {line!r}"
            ) from exc

    if not rows:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
        )

    labels = np.asarray(rows, dtype=np.float32)
    height, width = image_shape
    center_x = labels[:, 1] * width
    center_y = labels[:, 2] * height
    box_width = labels[:, 3] * width
    box_height = labels[:, 4] * height
    boxes = np.column_stack(
        (
            center_x - box_width / 2,
            center_y - box_height / 2,
            center_x + box_width / 2,
            center_y + box_height / 2,
        )
    )
    return labels[:, 0], boxes


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    """Return a stable, high-contrast BGR color for a class id."""
    palette = (
        (0, 255, 255),
        (255, 180, 0),
        (0, 200, 0),
        (0, 0, 255),
        (255, 0, 255),
        (255, 120, 0),
        (180, 0, 255),
        (0, 165, 255),
        (255, 0, 0),
        (80, 220, 120),
    )
    return palette[class_id % len(palette)]


def draw_labels(
    image: np.ndarray,
    classes: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    """Draw YOLO ground-truth boxes and class names on a copy of ``image``."""
    visual = image.copy()
    height, width = visual.shape[:2]

    for class_value, box in zip(classes, boxes):
        class_id = int(class_value)
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1 = int(np.clip(x1, 0, width - 1))
        y1 = int(np.clip(y1, 0, height - 1))
        x2 = int(np.clip(x2, 0, width - 1))
        y2 = int(np.clip(y2, 0, height - 1))
        color = _color_for_class(class_id)

        cv2.rectangle(visual, (x1, y1), (x2, y2), color, 2)
        class_name = (
            VISDRONE_CLASSES[class_id]
            if 0 <= class_id < len(VISDRONE_CLASSES)
            else f"class_{class_id}"
        )
        text = f"{class_id}: {class_name}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        text_y = max(y1, text_height + baseline + 2)
        cv2.rectangle(
            visual,
            (x1, text_y - text_height - baseline - 2),
            (min(x1 + text_width + 4, width - 1), text_y + 1),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            visual,
            text,
            (x1 + 2, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return visual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display YOLO ground-truth labels for an image."
    )
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument(
        "--label",
        type=Path,
        default=None,
        help="Label file; inferred from configs/paths.yaml when omitted.",
    )
    parser.add_argument("--output", type=Path, help="Optionally save the rendered image.")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a GUI window (useful together with --output).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_default_config()
    image_root = config.path_value("image_root")
    label_root = config.path_value("label_root")

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    label_path = (
        args.label.expanduser().resolve()
        if args.label is not None
        else image_to_label_path(image_path, image_root, label_root)
    )
    if not label_path.is_file():
        raise FileNotFoundError(f"Label not found: {label_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Could not read image: {image_path}")
    classes, boxes = read_yolo_labels(label_path, image.shape[:2])
    visual = draw_labels(image, classes, boxes)

    print(f"Image: {image_path}")
    print(f"Label: {label_path}")
    print(f"Objects: {len(boxes)}")

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), visual):
            raise OSError(f"Could not save image: {output_path}")
        print(f"Saved: {output_path}")

    if not args.no_show:
        cv2.imshow(f"Ground-truth labels - {image_path.name}", visual)
        print("Press any key in the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
