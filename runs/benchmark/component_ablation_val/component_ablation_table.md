# RL-SAHI component ablation - VisDrone2019-DET val

Each row uses the same detector, validation split, inference thresholds, crop acceptance settings, and evaluator. Only the DQN checkpoint changes.

Device: `cuda`. Checkpoint file name(s): `best.pt`.
YOLO full-image reference: AP=31.97, AP50=51.30, Recall-small@0.50=70.18.

| Variant | Spatial feature | Detection map | History | Outcome reward | Cost/overlap | Action mask | AP | Delta AP | AP50 | AP75 | Recall-small@0.50 | FP/image | Crops/image | Speed (img/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full RL-SAHI | Yes | Yes | Yes | Yes | Yes | Yes | 32.06 | +0.00 | 52.55 | 32.89 | 73.73 | 161.55 | 3.00 | 3.81 |
| w/o spatial feature | No | Yes | Yes | Yes | Yes | Yes | 31.69 | -0.36 | 51.47 | 32.72 | 68.50 | 129.01 | 3.00 | 3.84 |
| w/o detection map | Yes | No | Yes | Yes | Yes | Yes | 31.54 | -0.52 | 50.97 | 32.68 | 66.66 | 119.79 | 3.00 | 3.67 |
| w/o history | Yes | Yes | No | Yes | Yes | Yes | 31.81 | -0.24 | 51.57 | 32.94 | 69.01 | 131.16 | 3.00 | 3.82 |
| w/o outcome reward | Yes | Yes | Yes | No | Yes | Yes | 31.71 | -0.35 | 51.37 | 32.86 | 68.59 | 128.44 | 3.00 | 3.79 |
| w/o cost/overlap | Yes | Yes | Yes | Yes | No | Yes | 31.71 | -0.35 | 51.37 | 32.86 | 68.59 | 128.44 | 3.00 | 3.87 |
| w/o action mask | Yes | Yes | Yes | Yes | Yes | No | 31.98 | -0.08 | 52.02 | 32.99 | 70.99 | 142.45 | 2.24 | 4.86 |
