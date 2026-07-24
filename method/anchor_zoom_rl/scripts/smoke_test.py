from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT))

from tests.test_agent import test_double_dqn_update_accepts_action_masks
from tests.test_anchors import test_generate_anchors_clusters_detections_and_pads_with_grid
from tests.test_environment import test_action_maps_directly_to_roi_and_masks_selected_anchor
from tests.test_hard_regions import (
    test_hard_regions_include_missed_and_low_confidence_ground_truth,
    test_reward_adds_bonus_when_crop_recovers_hard_gt,
)
from tests.test_inference_smoke import test_inference_runs_one_direct_anchor_action
from tests.test_postprocess_metrics import (
    test_ap50_accumulator_reports_perfect_prediction,
    test_merge_is_class_aware_and_keeps_higher_score_duplicate,
)
from tests.test_replay import test_n_step_accumulator_flushes_terminal_suffixes
from tests.test_reward import test_empty_crop_is_rejected, test_reward_values_new_true_positive_and_crop_cost


def main() -> None:
    tests = [
        test_double_dqn_update_accepts_action_masks,
        test_generate_anchors_clusters_detections_and_pads_with_grid,
        test_action_maps_directly_to_roi_and_masks_selected_anchor,
        test_hard_regions_include_missed_and_low_confidence_ground_truth,
        test_reward_adds_bonus_when_crop_recovers_hard_gt,
        test_n_step_accumulator_flushes_terminal_suffixes,
        test_reward_values_new_true_positive_and_crop_cost,
        test_empty_crop_is_rejected,
        test_merge_is_class_aware_and_keeps_higher_score_duplicate,
        test_ap50_accumulator_reports_perfect_prediction,
    ]
    for test in tests:
        test()
        print(f"[smoke] PASS {test.__name__}")
    with TemporaryDirectory(prefix="anchor_zoom_rl_") as directory:
        test_inference_runs_one_direct_anchor_action(Path(directory))
    print("[smoke] PASS test_inference_runs_one_direct_anchor_action")
    print(f"[smoke] {len(tests) + 1} tests passed")


if __name__ == "__main__":
    main()
