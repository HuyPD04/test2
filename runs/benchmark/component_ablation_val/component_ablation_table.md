# RL-SAHI component ablation - VisDrone2019-DET val

Each row uses the same detector, validation split, inference thresholds, crop acceptance settings, and evaluator. Only the DQN checkpoint changes.

Device: `cuda`. Checkpoint file name(s): `best.pt`.
YOLO full-image reference: AP=27.13, AP50=43.82, Recall-small@0.50=54.45.

| Variant | Spatial feature | Detection map | History | Outcome reward | Cost/overlap | Action mask | AP | Delta AP | AP50 | AP75 | Recall-small@0.50 | FP/image | Crops/image | Speed (img/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full RL-SAHI | Yes | Yes | Yes | Yes | Yes | Yes | 32.01 | +0.00 | 54.01 | 32.07 | 69.48 | 106.72 | 3.00 | 6.11 |
| w/o spatial feature | No | Yes | Yes | Yes | Yes | Yes | 31.38 | -1.32 | 52.99 | 31.36 | 63.07 | 91.57 | 3.00 | 6.93 |
| w/o detection map | Yes | No | Yes | Yes | Yes | Yes | 31.37 | -1.34 | 52.99 | 31.36 | 63.00 | 91.40 | 3.00 | 6.86 |
| w/o history | Yes | Yes | No | Yes | Yes | Yes | 31.38 | -1.32 | 53 | 31.37 | 63.08 | 91.82 | 3.00 | 6.93 |
| w/o outcome reward | Yes | Yes | Yes | No | Yes | Yes | 31.39 | -1.32 | 53.01 | 31.39 | 63.09 | 91.65 | 3.00 | 6.80 |
| w/o cost/overlap | Yes | Yes | Yes | Yes | No | Yes | 31.39 | -1.32 | 53.01 | 31.39 | 63.09 | 91.65 | 3.00 | 6.90 |
| w/o action mask | Yes | Yes | Yes | Yes | Yes | No | 31.37 | -1.32 | 52.98 | 31.37 | 62.88 | 90.03 | 2.24 | 8.01 |
