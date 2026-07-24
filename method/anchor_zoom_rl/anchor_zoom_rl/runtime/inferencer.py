from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from ..config import MethodConfig
from ..core.anchors import generate_anchors
from ..core.postprocess import crop_reliability, crop_utility, merge_detections
from ..core.types import Detections
from ..rl.agent import DQNAgent
from .data import read_image
from .environment import AnchorZoomEnvironment
from .io import save_json, save_visdrone_predictions, save_visualization
from .prediction import DetectionRunner


@dataclass(slots=True)
class InferenceResult:
    image_path: Path
    detections: Detections
    full_detections: Detections
    attempted_rois: list[np.ndarray]
    accepted_rois: list[np.ndarray]
    actions: list[dict[str, Any]]
    timing: dict[str, float]
    stop_reason: str

    def metadata(self) -> dict[str, Any]:
        return {
            "image": str(self.image_path),
            "num_detections": len(self.detections),
            "num_attempted_crops": len(self.attempted_rois),
            "num_accepted_crops": len(self.accepted_rois),
            "attempted_rois": [roi.tolist() for roi in self.attempted_rois],
            "accepted_rois": [roi.tolist() for roi in self.accepted_rois],
            "actions": self.actions,
            "timing": self.timing,
            "stop_reason": self.stop_reason,
        }


class AnchorZoomInferencer:
    def __init__(
        self,
        cfg: MethodConfig,
        checkpoint: Path | None = None,
        runner: DetectionRunner | None = None,
    ) -> None:
        self.cfg = cfg
        self.runner = runner or DetectionRunner(cfg)
        self.agent = DQNAgent(
            state_dim=_state_dim(cfg),
            action_dim=cfg.action_count,
            cfg=cfg.train,
            device=cfg.detector.device,
        )
        checkpoint_path = Path(checkpoint or cfg.paths.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Policy checkpoint does not exist: {checkpoint_path}")
        payload = self.agent.load(checkpoint_path, load_optimizer=False)
        self.hardness_trained = int(
            payload.get("training_schema_version", 1)
        ) >= 3
        self.agent.online.eval()

    def infer_image(
        self,
        image_path: Path,
        split: str = "test",
        output_dir: Path | None = None,
        save: bool | None = None,
    ) -> InferenceResult:
        start = time.perf_counter()
        image_path = Path(image_path)
        image_read_start = time.perf_counter()
        image = read_image(image_path)
        image_read_ms = (time.perf_counter() - image_read_start) * 1000.0
        full_stage_start = time.perf_counter()
        full, full_ms, full_cache_hit = self.runner.full(
            image,
            image_path,
            split,
            use_cache=self.cfg.inference.cache_full_detections,
        )
        full_stage_ms = (time.perf_counter() - full_stage_start) * 1000.0
        anchor_start = time.perf_counter()
        anchors = generate_anchors(full, image.shape[:2], self.cfg.anchors)
        anchor_generation_ms = (time.perf_counter() - anchor_start) * 1000.0
        environment = AnchorZoomEnvironment(
            anchors, full, image.shape[:2], self.cfg.anchors, self.cfg.environment
        )
        predictions = merge_detections(
            Detections.empty(),
            full.filter_score(self.cfg.detector.output_confidence),
            self.cfg.detector.merge_iou,
            self.cfg.detector.max_detections,
            self.cfg.detector.cross_class_iou,
            self.cfg.detector.cross_class_ios,
            self.cfg.detector.cross_class_score_ratio,
            self.cfg.detector.cross_class_groups,
        )
        actions: list[dict[str, Any]] = []
        policy_ms = 0.0
        state_ms = 0.0
        crop_ms = 0.0
        crop_decision_ms = 0.0
        merge_ms = 0.0
        crop_cache_hits = 0
        stop_reason = "budget"

        while not environment.done:
            state_start = time.perf_counter()
            state = environment.state()
            mask = environment.action_mask()
            state_ms += (time.perf_counter() - state_start) * 1000.0
            policy_start = time.perf_counter()
            action, hard_probabilities = self.agent.select_action_with_hardness(
                state, mask
            )
            policy_ms += (time.perf_counter() - policy_start) * 1000.0
            if action == environment.stop_action:
                stop_reason = "policy_stop"
                valid_crop = np.asarray(mask, dtype=bool).copy()
                valid_crop[environment.stop_action] = False
                max_hardness = (
                    float(hard_probabilities[valid_crop].max())
                    if self.hardness_trained and valid_crop.any()
                    else None
                )
                actions.append(
                    {
                        "action": int(action),
                        "type": "stop",
                        "max_predicted_hardness": max_hardness,
                    }
                )
                break

            roi = environment.roi_for_action(action)
            overlap = environment.overlap_with_history(roi)
            crop, elapsed_ms, cache_hit = self.runner.crop(
                image,
                image_path,
                split,
                roi,
                use_cache=self.cfg.inference.cache_crop_detections,
            )
            crop_ms += elapsed_ms
            crop_cache_hits += int(cache_hit)
            decision_start = time.perf_counter()
            crop = crop.filter_score(self.cfg.detector.output_confidence)
            utility = crop_utility(
                crop,
                predictions,
                self.cfg.detector.duplicate_iou,
                self.cfg.reward.refinement_iou,
                self.cfg.reward.refinement_score_ratio,
                self.cfg.reward.refinement_utility_weight,
            )
            anchor_index, zoom_index = environment.decode_action(action)
            reliability = crop_reliability(
                crop,
                predictions,
                roi,
                anchor_score=environment.anchors[anchor_index].score,
                history_overlap=overlap,
                duplicate_iou=self.cfg.detector.duplicate_iou,
                refinement_iou=self.cfg.reward.refinement_iou,
                refinement_score_ratio=self.cfg.reward.refinement_score_ratio,
            )
            accepted = (
                len(crop) >= self.cfg.reward.min_crop_detections
                and utility >= self.cfg.reward.min_utility
                and reliability >= self.cfg.reward.min_reliability
            )
            crop_decision_ms += (time.perf_counter() - decision_start) * 1000.0
            if accepted:
                merge_start = time.perf_counter()
                predictions = merge_detections(
                    predictions,
                    crop,
                    self.cfg.detector.merge_iou,
                    self.cfg.detector.max_detections,
                    self.cfg.detector.cross_class_iou,
                    self.cfg.detector.cross_class_ios,
                    self.cfg.detector.cross_class_score_ratio,
                    self.cfg.detector.cross_class_groups,
                )
                merge_ms += (time.perf_counter() - merge_start) * 1000.0
            environment.record(
                action,
                roi,
                accepted,
                utility if accepted else 0.0,
            )
            actions.append(
                {
                    "action": int(action),
                    "type": "crop",
                    "anchor_index": int(anchor_index),
                    "zoom_index": int(zoom_index),
                    "zoom": float(self.cfg.anchors.zoom_bins[zoom_index]),
                    "roi": roi.tolist(),
                    "overlap": float(overlap),
                    "crop_detections": len(crop),
                    "utility": float(utility),
                    "reliability": float(reliability),
                    "predicted_hardness": (
                        float(hard_probabilities[action])
                        if self.hardness_trained
                        else None
                    ),
                    "accepted": bool(accepted),
                    "cache_hit": bool(cache_hit),
                }
            )

        predictions = predictions.filter_score(self.cfg.detector.output_confidence)
        total_ms = (time.perf_counter() - start) * 1000.0
        initial_state_ms = image_read_ms + full_stage_ms
        method_latency_ms = max(total_ms - initial_state_ms, 0.0)
        result = InferenceResult(
            image_path=image_path,
            detections=predictions,
            full_detections=full,
            attempted_rois=list(environment.attempted_rois),
            accepted_rois=list(environment.accepted_rois),
            actions=actions,
            timing={
                "image_read_ms": float(image_read_ms),
                "full_detection_ms": float(full_ms),
                "full_stage_ms": float(full_stage_ms),
                "initial_state_ms": float(initial_state_ms),
                "anchor_generation_ms": float(anchor_generation_ms),
                "state_ms": float(state_ms),
                "policy_ms": float(policy_ms),
                "crop_detection_ms": float(crop_ms),
                "crop_decision_ms": float(crop_decision_ms),
                "merge_ms": float(merge_ms),
                "method_latency_ms": float(method_latency_ms),
                "end_to_end_ms": float(total_ms),
                "full_cache_hit": float(full_cache_hit),
                "crop_cache_hits": float(crop_cache_hits),
            },
            stop_reason=stop_reason,
        )
        allow_save = True if save is None else bool(save)
        configured_output = (
            self.cfg.inference.save_predictions
            or self.cfg.inference.save_metadata
            or self.cfg.inference.save_visualization
        )
        if allow_save and configured_output:
            self._save_outputs(result, image, output_dir or self.cfg.paths.output_dir / "infer")
        return result

    def _save_outputs(
        self,
        result: InferenceResult,
        image: np.ndarray,
        output_dir: Path,
    ) -> None:
        output_dir = Path(output_dir)
        if self.cfg.inference.save_predictions:
            save_visdrone_predictions(
                output_dir / "predictions" / f"{result.image_path.stem}.txt",
                result.detections,
            )
        if self.cfg.inference.save_metadata:
            save_json(
                output_dir / "metadata" / f"{result.image_path.stem}.json",
                result.metadata(),
            )
        if self.cfg.inference.save_visualization:
            save_visualization(
                output_dir / "visualizations" / result.image_path.name,
                image,
                result.detections,
                result.attempted_rois,
                result.accepted_rois,
            )


def _state_dim(cfg: MethodConfig) -> int:
    from ..core.state import state_dimension

    return state_dimension(cfg.anchors)
