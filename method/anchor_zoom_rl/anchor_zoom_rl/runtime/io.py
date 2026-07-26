from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.types import Detections


def save_predictions(path: Path, detections: Detections) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    order = np.argsort(-detections.scores, kind="stable")
    for index in order:
        box = detections.boxes[index]
        score = detections.scores[index]
        class_id = detections.classes[index]
        x1, y1, x2, y2 = (float(value) for value in box)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{int(class_id)} {float(score):.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} 0")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def save_visdrone_predictions(path: Path, detections: Detections) -> None:
    save_predictions(path, detections)



def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def save_visualization(
    path: Path,
    image: np.ndarray,
    detections: Detections,
    attempted_rois: list[np.ndarray],
    accepted_rois: list[np.ndarray],
) -> None:
    canvas = image.copy()
    accepted_keys = {
        tuple(int(round(float(value))) for value in roi) for roi in accepted_rois
    }
    for roi in attempted_rois:
        coordinates = tuple(int(round(float(value))) for value in roi)
        color = (0, 200, 0) if coordinates in accepted_keys else (0, 165, 255)
        cv2.rectangle(canvas, coordinates[:2], coordinates[2:], color, 2)
    for box, score, class_id in zip(
        detections.boxes, detections.scores, detections.classes, strict=True
    ):
        x1, y1, x2, y2 = (int(round(float(value))) for value in box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 80, 40), 1)
        cv2.putText(
            canvas,
            f"{int(class_id)}:{float(score):.2f}",
            (x1, max(y1 - 3, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 80, 40),
            1,
            cv2.LINE_AA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)
