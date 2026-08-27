# RL-SAHI component ablation - VisDrone2019-DET val

Each row uses the same detector, validation split, inference thresholds, crop acceptance settings, and evaluator. Only the DQN checkpoint changes.
The full RL-SAHI baseline is benchmarked separately and is not included here.

Device: `cuda`. Checkpoint file name(s): `best.pt`.

| Variant | Detection map | History map | Hard-region target reward | Cost/overlap penalty | AP | AP50 | AP75 | Recall-small@0.50 | FP/image | Crops/image | Speed (img/s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w/o detection map | No | Yes | Yes | Yes | 31.86 | 51.66 | 32.73 | 76.43 | 161.47 | 2.99 | 5.65 |
| w/o history | Yes | No | Yes | Yes | 31.86 | 51.74 | 32.69 | 76.67 | 164.32 | 2.99 | 5.86 |
| w/o hard-region target reward | Yes | Yes | No | Yes | 31.89 | 51.76 | 32.69 | 76.66 | 163.77 | 3.00 | 5.73 |
| w/o cost/overlap | Yes | Yes | Yes | No | 31.97 | 51.80 | 32.86 | 76.16 | 159.42 | 2.98 | 5.34 |
