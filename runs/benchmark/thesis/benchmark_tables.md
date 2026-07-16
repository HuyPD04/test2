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
| YOLO11s only | local | 16.32 | 28.41 | 16.65 | 27.99 |
| RL-SAHI (proposed) | local | 16.64 | 29.46 | 16.74 | 5.78 |

> ASAHI paper values are reference-only: its TPH-YOLOv5 detector and hardware differ from the local YOLO11s pipeline, so the speed values are not controlled hardware comparisons.

## SOTA comparison - VisDrone2019-DET val

| Method | AP | AP50 | AP75 | Source |
|---|---:|---:|---:|---|
| ClusDet (ResNeXt-101, multi-scale) | 32.40 | 56.20 | 31.60 | ICCV 2019 |
| AdaZoom+CT (Cascade R-CNN, ResNeXt-101) | 40.33 | 66.94 | 41.77 | arXiv:2106.10409 |
| QueryDet (RetinaNet-50, CSQ) | 28.32 | 48.14 | 28.75 | CVPR 2022 |
| AD-Det* (ResNeXt-101) | 37.50 | 60.90 | 39.20 | Remote Sensing 2025 |
| TPH+ASAHI | 36.00 | 56.80 | 28.20 | arXiv:2604.19233 |
| RL-SAHI (proposed, YOLO11s) | 20.86 | 36.25 | 20.50 | local run |

## Internal detection diagnostics - local test run

| Method | Precision@0.50 | Recall@0.50 | Recall-small@0.50 | FP/image |
|---|---:|---:|---:|---:|
| YOLO11s only | 14.83 | 55.50 | 34.09 | 148.63 |
| SAHI budget 4 | 9.11 | 67.36 | 55.16 | 313.41 |
| SAHI budget 6 | 7.93 | 69.25 | 58.00 | 374.83 |
| SAHI budget 12 | 6.16 | 71.61 | 61.00 | 508.53 |
| SAHI budget 15 | 5.71 | 71.98 | 61.41 | 554.43 |
| RL-SAHI | 13.56 | 59.82 | 43.12 | 177.95 |

## Internal efficiency - local test run

| Method | Latency (ms/image) | Speed (img/s) | Slices/image | Detector calls/image | Effective GFLOPs |
|---|---:|---:|---:|---:|---:|
| YOLO11s only | 35.7 | 27.99 | 0.00 | 1.00 | 21.5 |
| SAHI budget 4 | 91.8 | 10.90 | 4.00 | 5.00 | 107.5 |
| SAHI budget 6 | 111.5 | 8.97 | 6.00 | 7.00 | 150.5 |
| SAHI budget 12 | 184.4 | 5.42 | 12.00 | 13.00 | 279.5 |
| SAHI budget 15 | 220.8 | 4.53 | 15.00 | 16.00 | 344.0 |
| RL-SAHI | 173.0 | 5.78 | 2.47 | 3.47 | 74.6 |

## Per-class AP - proposed method

| Class | Test AP | Test AP50 | Val AP | Val AP50 |
|---|---:|---:|---:|---:|
| 0: pedestrian | 10.90 | 28.65 | 18.43 | 43.09 |
| 1: people | 3.99 | 12.78 | 10.14 | 28.69 |
| 2: bicycle | 2.67 | 6.71 | 4.68 | 11.53 |
| 3: car | 44.18 | 72.44 | 54.35 | 80.63 |
| 4: van | 18.98 | 28.02 | 25.64 | 36.07 |
| 5: truck | 22.03 | 34.53 | 20.95 | 31.69 |
| 6: tricycle | 9.24 | 17.41 | 12.76 | 23.81 |
| 7: awning-tricycle | 6.86 | 12.24 | 6.14 | 9.78 |
| 8: bus | 35.93 | 51.55 | 37.60 | 53.69 |
| 9: motor | 11.61 | 30.26 | 17.88 | 43.54 |

## Protocol and provenance notes

- Local test split: VisDrone2019-DET test-dev (1,610 labeled images).
- Local val split: VisDrone2019-DET val (548 labeled images).
- Local AP is the mean over IoU 0.50:0.95; AP50 and AP75 use their single IoU thresholds.
- Effective GFLOPs follows Thesis.pdf: agent GFLOPs + detector GFLOPs x (1 + slices). YOLO11s is configured as 21.5 GFLOPs/pass; agent FLOPs are currently treated as negligible (0.0).
- `2602.07512v2.pdf` is ZoomDet, not ASAHI. ASAHI values come from `2604.19233v1.pdf`.
- Thesis.pdf retains all 10 classes, but its experimental-settings/results pages are accidentally replaced by MiniMedMind content; no result was copied from those corrupted pages.
- The current checkpoint predates the latest class/provenance schema and README marks it as stale; local RL-SAHI numbers are reproducible preliminary results, not a substitute for the planned retrain.
- `scripts/audit_splits.py` reports 24 shared sequence prefixes between train and val. The local val row must not be claimed as a final leak-free SOTA result until the split is repaired and the model is retrained.
