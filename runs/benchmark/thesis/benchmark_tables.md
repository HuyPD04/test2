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
| YOLO11s only | local | - | - | - | - |
| RL-SAHI (proposed) | local | - | - | - | - |

> ASAHI paper values are reference-only: its TPH-YOLOv5 detector and hardware differ from the local YOLO11s pipeline, so the speed values are not controlled hardware comparisons.

## SOTA comparison - VisDrone2019-DET val

| Method | AP | AP50 | AP75 | Source |
|---|---:|---:|---:|---|
| ClusDet (ResNeXt-101, multi-scale) | 32.40 | 56.20 | 31.60 | ICCV 2019 |
| AdaZoom+CT (Cascade R-CNN, ResNeXt-101) | 40.33 | 66.94 | 41.77 | arXiv:2106.10409 |
| QueryDet (RetinaNet-50, CSQ) | 28.32 | 48.14 | 28.75 | CVPR 2022 |
| AD-Det* (ResNeXt-101) | 37.50 | 60.90 | 39.20 | Remote Sensing 2025 |
| TPH+ASAHI | 36.00 | 56.80 | 28.20 | arXiv:2604.19233 |
| RL-SAHI (proposed, YOLO11s) | 18.68 | 30.29 | 19.32 | local run |

## Internal detection diagnostics - local test run

| Method | Precision@0.50 | Recall@0.50 | Recall-small@0.50 | FP/image |
|---|---:|---:|---:|---:|
| YOLO11s only | - | - | - | - |
| SAHI budget 4 | - | - | - | - |
| SAHI budget 6 | - | - | - | - |
| SAHI budget 12 | - | - | - | - |
| SAHI budget 15 | - | - | - | - |
| RL-SAHI | - | - | - | - |

## Internal efficiency - local test run

| Method | Latency (ms/image) | Speed (img/s) | Slices/image | Detector calls/image | Effective GFLOPs |
|---|---:|---:|---:|---:|---:|
| YOLO11s only | - | - | - | - | - |
| SAHI budget 4 | - | - | - | - | - |
| SAHI budget 6 | - | - | - | - | - |
| SAHI budget 12 | - | - | - | - | - |
| SAHI budget 15 | - | - | - | - | - |
| RL-SAHI | - | - | - | - | - |

## Per-class AP - proposed method

| Class | Test AP | Test AP50 | Val AP | Val AP50 |
|---|---:|---:|---:|---:|
| 0: pedestrian | - | - | 16.40 | 35.59 |
| 1: people | - | - | 7.92 | 20.45 |
| 2: bicycle | - | - | 3.49 | 7.46 |
| 3: car | - | - | 51.98 | 74.66 |
| 4: van | - | - | 23.85 | 32.72 |
| 5: truck | - | - | 18.37 | 26.11 |
| 6: tricycle | - | - | 9.92 | 16.57 |
| 7: awning-tricycle | - | - | 4.00 | 5.60 |
| 8: bus | - | - | 36.37 | 51.02 |
| 9: motor | - | - | 14.54 | 32.78 |

## Protocol and provenance notes

- Local test split: VisDrone2019-DET test-dev (1,610 labeled images).
- Local val split: VisDrone2019-DET val (548 labeled images).
- Local AP is the mean over IoU 0.50:0.95; AP50 and AP75 use their single IoU thresholds.
- Effective GFLOPs follows Thesis.pdf: agent GFLOPs + detector GFLOPs x (1 + slices). YOLO11s is configured as 21.5 GFLOPs/pass; agent FLOPs are currently treated as negligible (0.0).
- `2602.07512v2.pdf` is ZoomDet, not ASAHI. ASAHI values come from `2604.19233v1.pdf`.
- Thesis.pdf retains all 10 classes, but its experimental-settings/results pages are accidentally replaced by MiniMedMind content; no result was copied from those corrupted pages.
- The current checkpoint predates the latest class/provenance schema and README marks it as stale; local RL-SAHI numbers are reproducible preliminary results, not a substitute for the planned retrain.
- `scripts/audit_splits.py` reports 24 shared sequence prefixes between train and val. The local val row must not be claimed as a final leak-free SOTA result until the split is repaired and the model is retrained.
