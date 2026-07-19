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
| YOLO11s only | local | 25.08 | 41.82 | 25.71 | 14.67 |
| RL-SAHI (proposed) | local | 25.24 | 42.68 | 25.64 | 4.12 |

> ASAHI paper values are reference-only: its TPH-YOLOv5 detector and hardware differ from the local YOLO11s pipeline, so the speed values are not controlled hardware comparisons.

## SOTA comparison - VisDrone2019-DET val

| Method | AP | AP50 | AP75 | Source |
|---|---:|---:|---:|---|
| ClusDet (ResNeXt-101, multi-scale) | 32.40 | 56.20 | 31.60 | ICCV 2019 |
| AdaZoom+CT (Cascade R-CNN, ResNeXt-101) | 40.33 | 66.94 | 41.77 | arXiv:2106.10409 |
| QueryDet (RetinaNet-50, CSQ) | 28.32 | 48.14 | 28.75 | CVPR 2022 |
| AD-Det* (ResNeXt-101) | 37.50 | 60.90 | 39.20 | Remote Sensing 2025 |
| TPH+ASAHI | 36.00 | 56.80 | 28.20 | arXiv:2604.19233 |
| RL-SAHI (proposed, YOLO11s) | 32.06 | 52.55 | 32.89 | local run |

## Internal detection diagnostics - local test run

| Method | Precision@0.50 | Recall@0.50 | Recall-small@0.50 | FP/image |
|---|---:|---:|---:|---:|
| YOLO11s only | 21.28 | 69.37 | 54.39 | 119.74 |
| SAHI library | 14.77 | 71.87 | 56.72 | 193.51 |
| Fixed-grid Top-K 4 | - | - | - | - |
| Fixed-grid Top-K 6 | - | - | - | - |
| Fixed-grid Top-K 12 | - | - | - | - |
| Fixed-grid Top-K 15 | - | - | - | - |
| RL-SAHI | 24.77 | 69.86 | 56.98 | 98.97 |

## Internal efficiency - local test run

| Method | Latency (ms/image) | Speed (img/s) | Slices/image | Detector calls/image | Effective GFLOPs |
|---|---:|---:|---:|---:|---:|
| YOLO11s only | 68.2 | 14.67 | 0.00 | 1.00 | 21.5 |
| SAHI library | 184.8 | 5.41 | 6.11 | 7.11 | 152.9 |
| Fixed-grid Top-K 4 | - | - | - | - | - |
| Fixed-grid Top-K 6 | - | - | - | - | - |
| Fixed-grid Top-K 12 | - | - | - | - | - |
| Fixed-grid Top-K 15 | - | - | - | - | - |
| RL-SAHI | 242.6 | 4.12 | 3.00 | 4.00 | 86.0 |

## Per-class AP - proposed method

| Class | Test AP | Test AP50 | Val AP | Val AP50 |
|---|---:|---:|---:|---:|
| 0: pedestrian | 19.89 | 46.15 | 32.52 | 64.82 |
| 1: people | 10.00 | 26.75 | 21.16 | 50.28 |
| 2: bicycle | 10.25 | 21.79 | 16.53 | 33.97 |
| 3: car | 52.63 | 80.73 | 62.29 | 86.39 |
| 4: van | 29.05 | 40.45 | 37.11 | 50.51 |
| 5: truck | 32.05 | 48.57 | 32.38 | 47.27 |
| 6: tricycle | 17.85 | 31.00 | 24.94 | 41.63 |
| 7: awning-tricycle | 15.61 | 24.28 | 11.94 | 18.52 |
| 8: bus | 44.74 | 60.60 | 49.93 | 68.04 |
| 9: motor | 20.32 | 46.48 | 31.78 | 64.06 |

## Protocol and provenance notes

- Local test split: VisDrone2019-DET test-dev (1,610 labeled images).
- Local val split: VisDrone2019-DET val (548 labeled images).
- Local AP uses COCO-style 101-point interpolation over IoU 0.50:0.95 with maxDets=500; AP50 and AP75 use their single IoU thresholds. YOLO labels do not preserve VisDrone ignored-region metadata.
- Effective GFLOPs follows Thesis.pdf: agent GFLOPs + detector GFLOPs x (1 + slices). YOLO11s is configured as 21.5 GFLOPs/pass; agent FLOPs are currently treated as negligible (0.0).
- `2602.07512v2.pdf` is ZoomDet, not ASAHI. ASAHI values come from `2604.19233v1.pdf`.
- Thesis.pdf retains all 10 classes, but its experimental-settings/results pages are accidentally replaced by MiniMedMind content; no result was copied from those corrupted pages.
- The current checkpoint predates the latest class/provenance schema and README marks it as stale; local RL-SAHI numbers are reproducible preliminary results, not a substitute for the planned retrain.
- `scripts/audit_splits.py` reports 24 shared sequence prefixes between train and val. The local val row must not be claimed as a final leak-free SOTA result until the split is repaired and the model is retrained.
