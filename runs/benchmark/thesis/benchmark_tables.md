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
| YOLO11s only | local | 21.52 | 35.50 | 22.27 | 27.68 |
| RL-SAHI (proposed) | local | 23.06 | 38.51 | 23.52 | 5.88 |

> ASAHI paper values are reference-only: its TPH-YOLOv5 detector and hardware differ from the local YOLO11s pipeline, so the speed values are not controlled hardware comparisons.

## SOTA comparison - VisDrone2019-DET val

| Method | AP | AP50 | AP75 | Source |
|---|---:|---:|---:|---|
| ClusDet (ResNeXt-101, multi-scale) | 32.40 | 56.20 | 31.60 | ICCV 2019 |
| AdaZoom+CT (Cascade R-CNN, ResNeXt-101) | 40.33 | 66.94 | 41.77 | arXiv:2106.10409 |
| QueryDet (RetinaNet-50, CSQ) | 28.32 | 48.14 | 28.75 | CVPR 2022 |
| AD-Det* (ResNeXt-101) | 37.50 | 60.90 | 39.20 | Remote Sensing 2025 |
| TPH+ASAHI | 36.00 | 56.80 | 28.20 | arXiv:2604.19233 |
| RL-SAHI (proposed, YOLO11s) | 29.78 | 48.28 | 30.50 | local run |

## Internal detection diagnostics - local test run

| Method | Precision@0.50 | Recall@0.50 | Recall-small@0.50 | FP/image |
|---|---:|---:|---:|---:|
| YOLO11s only | 39.55 | 59.84 | 41.57 | 42.66 |
| SAHI library | 28.82 | 68.91 | 52.79 | 79.37 |
| Fixed-grid Top-K 4 | - | - | - | - |
| Fixed-grid Top-K 6 | - | - | - | - |
| Fixed-grid Top-K 12 | - | - | - | - |
| Fixed-grid Top-K 15 | - | - | - | - |
| RL-SAHI | 33.44 | 66.08 | 52.10 | 61.36 |

## Internal efficiency - local test run

| Method | Latency (ms/image) | Speed (img/s) | Slices/image | Detector calls/image | Effective GFLOPs |
|---|---:|---:|---:|---:|---:|
| YOLO11s only | 36.1 | 27.68 | 0.00 | 1.00 | 21.5 |
| SAHI library | 145.2 | 6.89 | 6.11 | 7.11 | 152.9 |
| Fixed-grid Top-K 4 | - | - | - | - | - |
| Fixed-grid Top-K 6 | - | - | - | - | - |
| Fixed-grid Top-K 12 | - | - | - | - | - |
| Fixed-grid Top-K 15 | - | - | - | - | - |
| RL-SAHI | 170.1 | 5.88 | 3.00 | 4.00 | 86.0 |

## Per-class AP - proposed method

| Class | Test AP | Test AP50 | Val AP | Val AP50 |
|---|---:|---:|---:|---:|
| 0: pedestrian | 17.98 | 42.27 | 30.21 | 61.28 |
| 1: people | 8.12 | 21.21 | 18.29 | 44.02 |
| 2: bicycle | 7.50 | 15.97 | 13.12 | 26.63 |
| 3: car | 50.95 | 77.82 | 61.24 | 84.36 |
| 4: van | 25.49 | 35.89 | 35.46 | 47.88 |
| 5: truck | 33.14 | 48.74 | 29.95 | 43.21 |
| 6: tricycle | 15.58 | 26.24 | 20.24 | 34.42 |
| 7: awning-tricycle | 11.25 | 17.67 | 10.15 | 16.18 |
| 8: bus | 41.82 | 56.65 | 50.58 | 66.05 |
| 9: motor | 18.76 | 42.66 | 28.61 | 58.74 |

## Protocol and provenance notes

- Local test split: VisDrone2019-DET test-dev (1,610 labeled images).
- Local val split: VisDrone2019-DET val (548 labeled images).
- Local AP uses COCO-style 101-point interpolation over IoU 0.50:0.95 with maxDets=500; AP50 and AP75 use their single IoU thresholds. YOLO labels do not preserve VisDrone ignored-region metadata.
- Effective GFLOPs follows Thesis.pdf: agent GFLOPs + detector GFLOPs x (1 + slices). YOLO11s is configured as 21.5 GFLOPs/pass; agent FLOPs are currently treated as negligible (0.0).
- `2602.07512v2.pdf` is ZoomDet, not ASAHI. ASAHI values come from `2604.19233v1.pdf`.
- Thesis.pdf retains all 10 classes, but its experimental-settings/results pages are accidentally replaced by MiniMedMind content; no result was copied from those corrupted pages.
- The current checkpoint predates the latest class/provenance schema and README marks it as stale; local RL-SAHI numbers are reproducible preliminary results, not a substitute for the planned retrain.
- `scripts/audit_splits.py` reports 24 shared sequence prefixes between train and val. The local val row must not be claimed as a final leak-free SOTA result until the split is repaired and the model is retrained.
