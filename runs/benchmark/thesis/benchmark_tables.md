# Thesis benchmark tables

AP and mAP denote the same COCO-style mean Average Precision in these tables. Local values use all 10 VisDrone classes and are shown as percentages.

## Internal comparison - VisDrone2019-DET test-dev

| Method | Source | AP | AP50 | AP75 | Speed (img/s) |
|---|---:|---:|---:|---:|---:|
| ASAHI (4 slices) | ASAHI paper | 23.9 | 38.7 | 17.1 | 5.19 |
| ASAHI (6 slices) | ASAHI paper | 28.5 | 41.6 | 22.0 | 4.98 |
| ASAHI (12 slices) | ASAHI paper | 29.3 | 41.9 | 22.8 | 2.98 |
| ASAHI (15 slices) | ASAHI paper | 27.2 | 40.9 | 21.3 | 2.39 |
| ASAHI (adaptive) | ASAHI paper | 30.4 | 45.6 | 25.2 | 4.88 |
| YOLO11s only | local | 16.62 | 28.70 | 16.97 | 20.53 |
| RL-SAHI (proposed) | local | 18.97 | 33.93 | 18.71 | 4.83 |

> ASAHI paper values are reference-only: its TPH-YOLOv5 detector and hardware differ from the local YOLO11s pipeline, so the speed values are not controlled hardware comparisons.

## SOTA comparison - VisDrone2019-DET val

| Method | AP | AP50 | AP75 | Source |
|---|---:|---:|---:|---|
| ClusDet (ResNeXt-101, multi-scale) | 32.40 | 56.20 | 31.60 | ICCV 2019 |
| AdaZoom+CT (Cascade R-CNN, ResNeXt-101) | 40.33 | 66.94 | 41.77 | arXiv:2106.10409 |
| QueryDet (RetinaNet-50, CSQ) | 28.32 | 48.14 | 28.75 | CVPR 2022 |
| AD-Det* (ResNeXt-101) | 37.50 | 60.90 | 39.20 | Remote Sensing 2025 |
| TPH+ASAHI | 36.00 | 56.80 | 28.20 | arXiv:2604.19233 |
| RL-SAHI (proposed, YOLO11s) | 25.82 | 44.21 | 25.59 | local run |

## Internal detection diagnostics - local test run

| Method | Precision@0.50 | Recall@0.50 | Recall-small@0.50 | FP/image |
|---|---:|---:|---:|---:|
| YOLO11s only | 14.90 | 54.37 | 31.62 | 144.83 |
| SAHI budget 4 | 15.78 | 66.93 | 51.57 | 166.66 |
| SAHI budget 6 | 15.04 | 68.76 | 53.88 | 181.24 |
| SAHI budget 12 | 13.76 | 71.15 | 56.48 | 208.05 |
| SAHI budget 15 | 13.42 | 71.54 | 56.81 | 215.32 |
| RL-SAHI | 16.87 | 62.57 | 46.12 | 143.81 |

## Internal efficiency - local test run

| Method | Latency (ms/image) | Speed (img/s) | Slices/image | Detector calls/image | Effective GFLOPs |
|---|---:|---:|---:|---:|---:|
| YOLO11s only | 48.7 | 20.53 | 0.00 | 1.00 | 21.5 |
| SAHI budget 4 | 88.9 | 11.25 | 4.00 | 5.00 | 107.5 |
| SAHI budget 6 | 96.0 | 10.41 | 6.00 | 7.00 | 150.5 |
| SAHI budget 12 | 123.5 | 8.10 | 12.00 | 13.00 | 279.5 |
| SAHI budget 15 | 140.0 | 7.14 | 15.00 | 16.00 | 344.0 |
| RL-SAHI | 207.2 | 4.83 | 3.00 | 4.00 | 86.0 |

## Per-class AP - proposed method

| Class | Test AP | Test AP50 | Val AP | Val AP50 |
|---|---:|---:|---:|---:|
| 0: pedestrian | 13.89 | 34.74 | 23.99 | 52.64 |
| 1: people | 5.84 | 17.44 | 14.75 | 39.09 |
| 2: bicycle | 4.86 | 11.11 | 7.18 | 17.32 |
| 3: car | 45.39 | 74.42 | 57.21 | 83.26 |
| 4: van | 20.88 | 30.98 | 32.08 | 44.88 |
| 5: truck | 25.44 | 41.08 | 27.46 | 41.51 |
| 6: tricycle | 11.21 | 21.63 | 17.57 | 31.94 |
| 7: awning-tricycle | 10.21 | 17.44 | 9.89 | 16.41 |
| 8: bus | 37.83 | 54.76 | 44.86 | 62.46 |
| 9: motor | 14.15 | 35.72 | 23.21 | 52.55 |

## Protocol and provenance notes

- Local test split: VisDrone2019-DET test-dev (1,610 labeled images).
- Local val split: VisDrone2019-DET val (548 labeled images).
- Local AP is the mean over IoU 0.50:0.95; AP50 and AP75 use their single IoU thresholds.
- Effective GFLOPs follows Thesis.pdf: agent GFLOPs + detector GFLOPs x (1 + slices). YOLO11s is configured as 21.5 GFLOPs/pass; agent FLOPs are currently treated as negligible (0.0).
- `2602.07512v2.pdf` is ZoomDet, not ASAHI. ASAHI values come from `2604.19233v1.pdf`.
- Thesis.pdf retains all 10 classes, but its experimental-settings/results pages are accidentally replaced by MiniMedMind content; no result was copied from those corrupted pages.
- The current checkpoint predates the latest class/provenance schema and README marks it as stale; local RL-SAHI numbers are reproducible preliminary results, not a substitute for the planned retrain.
- `scripts/audit_splits.py` reports 24 shared sequence prefixes between train and val. The local val row must not be claimed as a final leak-free SOTA result until the split is repaired and the model is retrained.
