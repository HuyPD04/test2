# RL-SAHI ablation - VisDrone2019-DET val

All variants use the same detector, DQN checkpoint, validation images, confidence thresholds, merge settings, and evaluator. The STOP gate is `require_stop_for_acceptance`; regular crop utility checks remain enabled in every RL variant.

Execution device for the reported speed values: `cpu`. Accuracy metrics are comparable across these rows; do not compare this speed column with T4 paper results.

| Configuration | ROI pre-filter | STOP gate | AP | AP50 | AP75 | Recall-small@0.50 | FP/image | Crops/image | Speed (img/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11s full-image | No | No | 20.91 | 35.03 | 21.22 | 40.50 | 190.59 | 0.00 | 8.82 |
| RL slicing | No | No | 25.81 | 44.19 | 25.69 | 68.07 | 253.68 | 4.00 | 1.38 |
| RL slicing + ROI pre-filter | Yes | No | 25.47 | 43.49 | 25.41 | 65.89 | 238.09 | 3.00 | 1.71 |
| RL slicing + ROI pre-filter + STOP gate | Yes | Yes | 25.46 | 43.45 | 25.40 | 65.78 | 236.85 | 2.94 | 1.62 |
