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
| YOLO11s only | local | 14.38 | 23.39 | 15.44 | 28.05 |
| RL-SAHI (proposed) | local | 14.97 | 24.79 | 15.80 | 3.83 |

> ASAHI paper values are reference-only: its TPH-YOLOv5 detector and hardware differ from the local YOLO11s pipeline, so the speed values are not controlled hardware comparisons.

## SOTA comparison - VisDrone2019-DET val

| Method | AP | AP50 | AP75 | Source |
|---|---:|---:|---:|---|
| ClusDet (ResNeXt-101, multi-scale) | 32.40 | 56.20 | 31.60 | ICCV 2019 |
| AdaZoom+CT (Cascade R-CNN, ResNeXt-101) | 40.33 | 66.94 | 41.77 | arXiv:2106.10409 |
| QueryDet (RetinaNet-50, CSQ) | 28.32 | 48.14 | 28.75 | CVPR 2022 |
| AD-Det* (ResNeXt-101) | 37.50 | 60.90 | 39.20 | Remote Sensing 2025 |
| TPH+ASAHI | 36.00 | 56.80 | 28.20 | arXiv:2604.19233 |
| RL-SAHI (proposed, YOLO11s) | 18.67 | 30.20 | 19.32 | local run |

## Internal detection diagnostics - local test run

| Method | Precision@0.50 | Recall@0.50 | Recall-small@0.50 | FP/image |
|---|---:|---:|---:|---:|
| YOLO11s only | 67.00 | 41.98 | 19.07 | 9.65 |
| SAHI budget 4 | 56.15 | 50.93 | 33.70 | 18.55 |
| SAHI budget 6 | 53.00 | 52.24 | 35.50 | 21.61 |
| SAHI budget 12 | 47.12 | 53.81 | 37.14 | 28.17 |
| SAHI budget 15 | 45.45 | 54.07 | 37.30 | 30.27 |
| RL-SAHI | 61.64 | 46.46 | 27.05 | 13.49 |

## Internal efficiency - local test run

| Method | Latency (ms/image) | Speed (img/s) | Slices/image | Detector calls/image | Effective GFLOPs |
|---|---:|---:|---:|---:|---:|
| YOLO11s only | 35.7 | 28.05 | 0.00 | 1.00 | 21.5 |
| SAHI budget 4 | 90.9 | 11.00 | 4.00 | 5.00 | 107.5 |
| SAHI budget 6 | 111.4 | 8.97 | 6.00 | 7.00 | 150.5 |
| SAHI budget 12 | 189.4 | 5.28 | 12.00 | 13.00 | 279.5 |
| SAHI budget 15 | 233.6 | 4.28 | 15.00 | 16.00 | 344.0 |
| RL-SAHI | 261.3 | 3.83 | 3.18 | 4.18 | 89.9 |

## Per-class AP - proposed method

| Class | Test AP | Test AP50 | Val AP | Val AP50 |
|---|---:|---:|---:|---:|
| 0: pedestrian | 9.76 | 24.11 | 16.38 | 35.47 |
| 1: people | 2.75 | 7.97 | 7.92 | 20.32 |
| 2: bicycle | 2.15 | 4.71 | 3.54 | 7.50 |
| 3: car | 42.24 | 67.21 | 51.99 | 74.64 |
| 4: van | 17.63 | 25.28 | 24.11 | 32.86 |
| 5: truck | 20.02 | 30.01 | 18.27 | 25.92 |
| 6: tricycle | 7.70 | 13.38 | 10.16 | 17.13 |
| 7: awning-tricycle | 4.62 | 6.91 | 4.00 | 5.58 |
| 8: bus | 33.31 | 45.94 | 35.97 | 50.23 |
| 9: motor | 9.49 | 22.34 | 14.41 | 32.39 |

## Protocol and provenance notes

- Local test split: VisDrone2019-DET test-dev (1,610 labeled images).
- Local val split: VisDrone2019-DET val (548 labeled images).
- Local AP is the mean over IoU 0.50:0.95; AP50 and AP75 use their single IoU thresholds.
- Effective GFLOPs follows Thesis.pdf: agent GFLOPs + detector GFLOPs x (1 + slices). YOLO11s is configured as 21.5 GFLOPs/pass; agent FLOPs are currently treated as negligible (0.0).
- `2602.07512v2.pdf` is ZoomDet, not ASAHI. ASAHI values come from `2604.19233v1.pdf`.
- Thesis.pdf retains all 10 classes, but its experimental-settings/results pages are accidentally replaced by MiniMedMind content; no result was copied from those corrupted pages.
- The current checkpoint predates the latest class/provenance schema and README marks it as stale; local RL-SAHI numbers are reproducible preliminary results, not a substitute for the planned retrain.
- `scripts/audit_splits.py` reports 24 shared sequence prefixes between train and val. The local val row must not be claimed as a final leak-free SOTA result until the split is repaired and the model is retrained.
