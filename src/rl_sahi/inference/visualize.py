from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rl_sahi.common.boxes import as_boxes
from rl_sahi.common.data import read_image


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

def draw_boxes(image: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int], classes: np.ndarray | None = None, thickness: int = 1) -> None:
    for i, box in enumerate(as_boxes(boxes)):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        if classes is not None:
            cls_id = int(classes[i])
            cls_name = VISDRONE_CLASSES[cls_id] if 0 <= cls_id < len(VISDRONE_CLASSES) else str(cls_id)
            cv2.putText(image, cls_name, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness)


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    sources: np.ndarray,
    classes: np.ndarray | None = None,
    full_color: tuple[int, int, int] = (0, 190, 0),
    slice_color: tuple[int, int, int] = (255, 120, 0),
) -> None:
    boxes = as_boxes(boxes)
    sources = np.asarray(sources, dtype=np.int32).reshape(-1)
    if len(boxes) == 0:
        return
    classes_full = classes[sources == 0] if classes is not None else None
    classes_slice = classes[sources != 0] if classes is not None else None
    draw_boxes(image, boxes[sources == 0], full_color, classes_full, thickness=1)
    draw_boxes(image, boxes[sources != 0], slice_color, classes_slice, thickness=1)


def save_inference_visual(
    image_path: Path,
    boxes: np.ndarray,
    sources: np.ndarray,
    classes: np.ndarray,
    accepted_rois: np.ndarray,
    rejected_rois: np.ndarray,
    out_path: Path,
) -> None:
    image = read_image(image_path)
    draw_detections(image, boxes, sources, classes)
    # draw_boxes(image, rejected_rois, (0, 165, 255), thickness=2)
    draw_boxes(image, accepted_rois, (0, 0, 255), thickness=2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)
