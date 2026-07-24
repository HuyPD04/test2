from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))

from tests.test_agent import (
    test_auxiliary_hard_action_loss_updates_head,
    test_double_dqn_update_accepts_action_masks,
)
from tests.test_anchors import test_generate_anchors_clusters_detections_and_pads_with_grid
from tests.test_environment import test_action_maps_directly_to_roi_and_masks_selected_anchor
from tests.test_hard_regions import (
    test_hard_regions_include_missed_and_low_confidence_ground_truth,
    test_hard_action_supervision_and_dynamic_stop_penalty,
    test_reward_adds_bonus_when_crop_recovers_hard_gt,
)
from tests.test_inference_smoke import test_inference_runs_one_direct_anchor_action
from tests.test_postprocess_metrics import (
    test_ap50_accumulator_reports_perfect_prediction,
    test_ambiguous_overlap_below_merge_iou_is_not_refinement,
    test_cross_class_cleanup_suppresses_lower_score_duplicate,
    test_crop_refinement_has_utility_and_reliability,
    test_crop_reliability_penalizes_boundary_noise,
    test_merge_is_class_aware_and_keeps_higher_score_duplicate,
)
from tests.test_replay import (
    test_n_step_accumulator_flushes_terminal_suffixes,
    test_n_step_keeps_auxiliary_target_from_first_state,
)
from tests.test_reward import (
    test_empty_crop_is_rejected,
    test_low_reliability_crop_is_rejected,
    test_reward_values_new_true_positive_and_crop_cost,
)
from tests.test_sampling import (
    test_shuffled_epoch_sampler_visits_every_image_once,
    test_stratified_sample_covers_sequences_before_repeating,
)
from tests.test_training_smoke import (
    test_training_writes_schema3_metrics_and_best_checkpoints,
)


def main() -> None:
    tests = [
        test_double_dqn_update_accepts_action_masks,
        test_auxiliary_hard_action_loss_updates_head,
        test_generate_anchors_clusters_detections_and_pads_with_grid,
        test_action_maps_directly_to_roi_and_masks_selected_anchor,
        test_hard_regions_include_missed_and_low_confidence_ground_truth,
        test_hard_action_supervision_and_dynamic_stop_penalty,
        test_reward_adds_bonus_when_crop_recovers_hard_gt,
        test_n_step_accumulator_flushes_terminal_suffixes,
        test_n_step_keeps_auxiliary_target_from_first_state,
        test_reward_values_new_true_positive_and_crop_cost,
        test_empty_crop_is_rejected,
        test_low_reliability_crop_is_rejected,
        test_merge_is_class_aware_and_keeps_higher_score_duplicate,
        test_cross_class_cleanup_suppresses_lower_score_duplicate,
        test_crop_reliability_penalizes_boundary_noise,
        test_crop_refinement_has_utility_and_reliability,
        test_ambiguous_overlap_below_merge_iou_is_not_refinement,
        test_ap50_accumulator_reports_perfect_prediction,
        test_stratified_sample_covers_sequences_before_repeating,
        test_shuffled_epoch_sampler_visits_every_image_once,
    ]
    for test in tests:
        test()
        print(f"[smoke] PASS {test.__name__}")
    with TemporaryDirectory(prefix="anchor_zoom_rl_") as directory:
        root = Path(directory)
        infer_root = root / "infer"
        train_root = root / "train"
        infer_root.mkdir()
        train_root.mkdir()
        test_inference_runs_one_direct_anchor_action(infer_root)
        test_training_writes_schema3_metrics_and_best_checkpoints(train_root)
    print("[smoke] PASS test_inference_runs_one_direct_anchor_action")
    print("[smoke] PASS test_training_writes_schema3_metrics_and_best_checkpoints")
    print(f"[smoke] {len(tests) + 2} tests passed")


if __name__ == "__main__":
    main()
