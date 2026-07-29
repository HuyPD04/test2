# RL-SAHI component ablation

All five runs use the same base configuration, data, cached detector state, crop-outcome reward, action mask, and training budget. Train them from scratch with separate output directories:

```powershell
conda run -n doan python scripts/train.py --config configs/ablations/components/full.yaml --split train --no-resume
conda run -n doan python scripts/train.py --config configs/ablations/components/no_history.yaml --split train --no-resume
conda run -n doan python scripts/train.py --config configs/ablations/components/no_detection_map.yaml --split train --no-resume
conda run -n doan python scripts/train.py --config configs/ablations/components/no_hard_region_reward.yaml --split train --no-resume
conda run -n doan python scripts/train.py --config configs/ablations/components/no_cost_overlap.yaml --split train --no-resume
```

Then benchmark all trained checkpoints on the same validation split:

```powershell
conda run -n doan python scripts/run_component_ablation.py --split val
```

`no_hard_region_reward` deliberately still loads the same hard-region cache so that samples and cache metadata are unchanged, but the environment exposes no hard boxes and applies no hard-region target or coverage reward.
