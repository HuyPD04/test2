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
| YOLO11s only | local | 14.59 | 23.47 | 15.68 | 9.36 |
| RL-SAHI (proposed) | local | 17.58 | 29.44 | 18.26 | 1.14 |

> ASAHI paper values are reference-only: its TPH-YOLOv5 detector and hardware differ from the local YOLO11s pipeline, so the speed values are not controlled hardware comparisons.

## SOTA comparison - VisDrone2019-DET val

| Method | AP | AP50 | AP75 | Source |
|---|---:|---:|---:|---|
| ClusDet (ResNeXt-101, multi-scale) | 32.40 | 56.20 | 31.60 | ICCV 2019 |
| AdaZoom+CT (Cascade R-CNN, ResNeXt-101) | 40.33 | 66.94 | 41.77 | arXiv:2106.10409 |
| QueryDet (RetinaNet-50, CSQ) | 28.32 | 48.14 | 28.75 | CVPR 2022 |
| AD-Det* (ResNeXt-101) | 37.50 | 60.90 | 39.20 | Remote Sensing 2025 |
| TPH+ASAHI | 36.00 | 56.80 | 28.20 | arXiv:2604.19233 |
| RL-SAHI (proposed, YOLO11s) | 23.37 | 37.50 | 24.46 | local run |

## Internal detection diagnostics - local test run

| Method | Precision@0.50 | Recall@0.50 | Recall-small@0.50 | FP/image |
|---|---:|---:|---:|---:|
| YOLO11s only | 68.90 | 40.72 | 16.35 | 8.57 |
| SAHI budget 4 | 63.75 | 54.96 | 37.37 | 14.58 |
| SAHI budget 6 | 61.99 | 57.07 | 39.92 | 16.32 |
| SAHI budget 12 | 58.22 | 59.51 | 42.27 | 19.92 |
| SAHI budget 15 | 57.18 | 59.91 | 42.53 | 20.93 |
| RL-SAHI | 64.90 | 51.01 | 32.17 | 12.87 |

## Internal efficiency - local test run

| Method | Latency (ms/image) | Speed (img/s) | Slices/image | Detector calls/image | Effective GFLOPs |
|---|---:|---:|---:|---:|---:|
| YOLO11s only | 106.8 | 9.36 | 0.00 | 1.00 | 21.5 |
| SAHI budget 4 | 583.6 | 1.71 | 4.00 | 5.00 | 107.5 |
| SAHI budget 6 | 806.9 | 1.24 | 6.00 | 7.00 | 150.5 |
| SAHI budget 12 | 1492.8 | 0.67 | 12.00 | 13.00 | 279.5 |
| SAHI budget 15 | 1876.9 | 0.53 | 15.00 | 16.00 | 344.0 |
| RL-SAHI | 877.3 | 1.14 | 4.73 | 5.73 | 123.2 |

## Per-class AP - proposed method

| Class | Test AP | Test AP50 | Val AP | Val AP50 |
|---|---:|---:|---:|---:|
| 0: pedestrian | 13.14 | 30.51 | 22.34 | 45.31 |
| 1: people | 4.96 | 13.45 | 11.84 | 28.94 |
| 2: bicycle | 4.14 | 8.51 | 5.93 | 12.41 |
| 3: car | 43.87 | 70.10 | 54.62 | 78.26 |
| 4: van | 20.43 | 29.50 | 29.84 | 41.40 |
| 5: truck | 23.39 | 35.90 | 24.19 | 34.41 |
| 6: tricycle | 9.53 | 16.12 | 13.40 | 23.22 |
| 7: awning-tricycle | 7.99 | 12.66 | 8.13 | 11.53 |
| 8: bus | 35.79 | 49.26 | 43.73 | 58.70 |
| 9: motor | 12.59 | 28.36 | 19.73 | 40.79 |

## Protocol and provenance notes

- Local test split: VisDrone2019-DET test-dev (1,610 labeled images).
- Local val split: VisDrone2019-DET val (548 labeled images).
- Local AP is the mean over IoU 0.50:0.95; AP50 and AP75 use their single IoU thresholds.
- Effective GFLOPs follows Thesis.pdf: agent GFLOPs + detector GFLOPs x (1 + slices). YOLO11s is configured as 21.5 GFLOPs/pass; agent FLOPs are currently treated as negligible (0.0).
- `2602.07512v2.pdf` is ZoomDet, not ASAHI. ASAHI values come from `2604.19233v1.pdf`.
- Thesis.pdf retains all 10 classes, but its experimental-settings/results pages are accidentally replaced by MiniMedMind content; no result was copied from those corrupted pages.
- The current checkpoint predates the latest class/provenance schema and README marks it as stale; local RL-SAHI numbers are reproducible preliminary results, not a substitute for the planned retrain.
- `scripts/audit_splits.py` reports 24 shared sequence prefixes between train and val. The local val row must not be claimed as a final leak-free SOTA result until the split is repaired and the model is retrained.
