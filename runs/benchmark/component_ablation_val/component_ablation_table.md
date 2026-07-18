# RL-SAHI component ablation - VisDrone2019-DET val

Each row uses the same detector, validation split, inference thresholds, crop acceptance settings, and evaluator. Only the DQN checkpoint changes.

Device: `cpu`. Checkpoint file name(s): `best.pt`.
YOLO full-image reference: AP=20.91, AP50=35.03, Recall-small@0.50=40.50.

| Variant | Spatial feature | Detection map | History | Outcome reward | Cost/overlap | Action mask | AP | Delta AP | AP50 | AP75 | Recall-small@0.50 | FP/image | Crops/image | Speed (img/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full RL-SAHI | Yes | Yes | Yes | Yes | Yes | Yes | 25.47 | +0.00 | 43.49 | 25.41 | 65.89 | 238.09 | 3.00 | 2.42 |
| w/o spatial feature | No | Yes | Yes | Yes | Yes | Yes | 21.68 | -3.78 | 36.94 | 21.69 | 45.15 | 170.94 | 3.00 | 2.51 |
| w/o detection map | Yes | No | Yes | Yes | Yes | Yes | 21.23 | -4.24 | 36.04 | 21.30 | 43.09 | 164.95 | 3.00 | 2.40 |
| w/o history | Yes | Yes | No | Yes | Yes | Yes | 22.32 | -3.15 | 37.88 | 22.46 | 48.79 | 183.86 | 3.00 | 2.38 |
| w/o outcome reward | Yes | Yes | Yes | No | Yes | Yes | 21.97 | -3.50 | 37.35 | 22.04 | 46.81 | 175.79 | 3.00 | 2.41 |
| w/o cost/overlap | Yes | Yes | Yes | Yes | No | Yes | 21.97 | -3.50 | 37.35 | 22.04 | 46.81 | 175.79 | 3.00 | 2.41 |
| w/o action mask | Yes | Yes | Yes | Yes | Yes | No | 22.98 | -2.49 | 39.42 | 22.87 | 54.34 | 198.16 | 2.25 | 3.17 |
