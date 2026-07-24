from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _empty_boxes() -> np.ndarray:
    return np.zeros((0, 4), dtype=np.float32)


def _empty_vector(dtype: np.dtype = np.float32) -> np.ndarray:
    return np.zeros((0,), dtype=dtype)


@dataclass(slots=True)
class Detections:
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        self.classes = np.asarray(self.classes, dtype=np.int64).reshape(-1)
        if not (len(self.boxes) == len(self.scores) == len(self.classes)):
            raise ValueError("boxes, scores, and classes must have the same length")

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    @classmethod
    def empty(cls) -> "Detections":
        return cls(_empty_boxes(), _empty_vector(), _empty_vector(np.int64))

    @classmethod
    def concatenate(cls, *items: "Detections") -> "Detections":
        valid = [item for item in items if len(item)]
        if not valid:
            return cls.empty()
        return cls(
            np.concatenate([item.boxes for item in valid], axis=0),
            np.concatenate([item.scores for item in valid], axis=0),
            np.concatenate([item.classes for item in valid], axis=0),
        )

    def take(self, indices: np.ndarray | list[int]) -> "Detections":
        idx = np.asarray(indices, dtype=np.int64)
        return Detections(self.boxes[idx], self.scores[idx], self.classes[idx])

    def filter_score(self, threshold: float) -> "Detections":
        return self.take(np.flatnonzero(self.scores >= float(threshold)))

    def filter_classes(self, target_classes: tuple[int, ...]) -> "Detections":
        if not target_classes:
            return self
        keep = np.isin(self.classes, np.asarray(target_classes, dtype=np.int64))
        return self.take(np.flatnonzero(keep))


@dataclass(slots=True, frozen=True)
class Anchor:
    cx: float
    cy: float
    width: float
    height: float
    score: float
    count: int
    mean_conf: float
    max_conf: float
    small_fraction: float
    mean_area_ratio: float
    source: str = "detections"


@dataclass(slots=True)
class StepOutcome:
    accepted: bool
    reward: float
    utility: float
    tp_gain: int
    hard_tp_gain: int
    fp_gain: int
    small_tp_gain: int
    hard_coverage_gain: int
    reason: str
