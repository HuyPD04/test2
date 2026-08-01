# RL-SAHI component ablation - VisDrone2019-DET val

Each row uses the same detector, validation split, inference thresholds, crop acceptance settings, and evaluator. Only the DQN checkpoint changes.
The full RL-SAHI baseline is benchmarked separately and is not included here.

Device: `cuda`. Checkpoint file name(s): `best.pt`.

| Variant | Detection map | History map | Hard-region target reward | Cost/overlap penalty | AP | AP50 | AP75 | Recall-small@0.50 | FP/image | Crops/image | Speed (img/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o detection map | No | Yes | Yes | Yes | 27.57 | 45.75 | 27.83 | 69.28 | 175.37 | 2.99 | 3.39 |
| w/o history | Yes | No | Yes | Yes | 27.60 | 45.84 | 27.85 | 69.46 | 177.05 | 2.99 | 3.36 |
| w/o hard-region target reward | Yes | Yes | No | Yes | 27.63 | 45.84 | 27.89 | 69.38 | 176.69 | 2.99 | 3.37 |
| w/o cost/overlap | Yes | Yes | Yes | No | 27.60 | 45.83 | 27.91 | 68.88 | 170.76 | 2.97 | 3.54 |
