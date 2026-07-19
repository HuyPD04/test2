from __future__ import annotations

from dataclasses import dataclass, field

from rl_sahi.common.class_mapping import ClassMapping


@dataclass(slots=True)
class InferenceConfig:
    full_imgsz: int = 640
    slice_imgsz: int = 640
    full_conf: float = 0.01
    output_conf: float = 0.3
    iou: float = 0.7
    merge_iou: float = 0.5
    max_det: int = 3000
    device: str | None = None
    policy_device: str | None = None
    feature_layers: tuple[int, ...] = (10,)
    min_slice_detections: int = 1
    min_slice_utility: float = 0.5
    min_new_detection_score: float = 0.45
    duplicate_iou: float = 0.5
    max_slice_attempts: int = 0
    roi_prefilter_enabled: bool = False
    roi_prefilter_topk: int = 3
    crop_batch_size: int = 1
    max_consecutive_rejections: int = 0
    target_classes: tuple[int, ...] = (0, 2, 3, 5, 8, 9)
    require_stop_for_acceptance: bool = True
    save_predictions: bool = True
    save_metadata: bool = True
    save_visualization: bool = False
    batched_inference: bool = False
    use_wbf: bool = False
    class_mapping: ClassMapping = field(default_factory=ClassMapping)

