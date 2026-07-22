# RL-SAHI

RL-SAHI là pipeline phát hiện vật thể nhỏ trên ảnh VisDrone, kết hợp YOLO11s full-image với policy Dueling Double DQN để chọn các ROI cần chạy lại detector ở độ phân giải crop.

Pipeline tối ưu đồng thời recall vật thể nhỏ, số crop, false positive và latency. Policy hiện dùng 11 action để di chuyển/zoom ROI và `STOP`; outer inference dùng budget thích nghi và có thể dừng sớm sau một chuỗi crop bị từ chối.

## Trạng thái hiện tại

Cache schema, class scope, reward, acceptance gate và benchmark đã thay đổi. Checkpoint/cache cũ được giữ lại để tham khảo nhưng không nên dùng để tiếp tục train.

Trước lần retrain tiếp theo cần:

1. Tách lại train/val theo sequence prefix để không có cùng sequence ở hai split.
2. Rebuild detection cache và hard-region cache bằng weights/config hiện tại.
3. Chạy cache checker.
4. Train mới với resume tắt.
5. Chạy benchmark stratified hoặc full test và cập nhật kết quả.

## Class scope

Sáu lớp mục tiêu được khai báo một lần trong `configs/default.yaml`:

```yaml
classes:
  target_classes: [0, 2, 3, 5, 8, 9]
```

YOLO weights hiện tại đã được train trực tiếp trên 10 lớp VisDrone, vì vậy không dùng COCO-to-VisDrone class mapping. Train, hard-region mining, inference và benchmark đều đọc cùng class scope trên. Thiếu hoặc trùng class id sẽ làm pipeline dừng sớm.

## Cài đặt

Khuyến nghị Python 3.11 và PyTorch có CUDA:

```powershell
conda create -n rl-sahi python=3.11
conda activate rl-sahi
pip install -r requirements.txt
```

Nếu không có CUDA, đổi `device` trong config sang `cpu` hoặc chuỗi rỗng. Benchmark luôn ghi effective device vào manifest.

## Dữ liệu

Dataset dùng YOLO format:

```text
data/raw/images/train/*.jpg
data/raw/images/val/*.jpg
data/raw/images/test/*.jpg
data/raw/labels/train/*.txt
data/raw/labels/val/*.txt
data/raw/labels/test/*.txt
```

Tên ảnh được xem là có sequence id ở phần trước dấu gạch dưới đầu tiên. Training mặc định từ chối chạy nếu train và val có sequence id trùng nhau. Chỉ đặt `train.allow_sequence_overlap: true` khi đã xác nhận prefix không phải sequence id.

Kiểm tra split:

```powershell
python scripts\audit_splits.py
```

## Workflow retrain

### 1. Rebuild detection cache

Cache mới lưu version, config và SHA-256 của weights. Rebuild cả ba split sau khi thay weights hoặc detection/state config:

```powershell
python scripts\detect.py --split train --overwrite
python scripts\detect.py --split val --overwrite
python scripts\detect.py --split test --overwrite
```

### 2. Rebuild hard-region cache

Hard-region cache mới lưu class scope, class mapping, detector fingerprint và các threshold mining:

```powershell
python scripts\hard_region.py --split train --overwrite
python scripts\hard_region.py --split val --overwrite
```

Script tự rebuild file stale ngay cả khi không truyền `--overwrite`; flag này vẫn được khuyến nghị sau thay đổi lớn.

### 3. Kiểm tra cache

```powershell
python tests\check_caches.py
```

Có thể kiểm tra nhanh một split:

```powershell
python tests\check_caches.py --split train --limit 100
```

Checker trả exit code khác 0 nếu thiếu hoặc stale cache.

### 4. Train mới

```powershell
python scripts\train.py --split train --no-resume
```

`train.resume` mặc định là `false`. Resume checkpoint lưu detection metadata, target classes, class mapping, inference config và benchmark config; resume sẽ bị từ chối nếu provenance không khớp.

Checkpoint:

```text
runs/dqn/best.pt
runs/dqn/last.pt
runs/dqn/resume.pt
```

`best.pt` được chọn bằng benchmark validation gồm AP50, small recall, FP cost và crop cost. Benchmark images được lấy stratified theo sequence thay vì lấy N file đầu theo tên.

## Inference

```powershell
python scripts\infer.py --split test --limit 20
python scripts\infer.py --image image.png --visualize
```

Acceptance gate mới:

- Novel detection count phải đạt `min_slice_detections`.
- Novel max confidence phải đạt `min_new_detection_score`.
- Một novel detection đủ confidence được nhận mà không cần tổng utility đạt 0.8.
- Crop có nhiều novel detections vẫn phải đạt `min_slice_utility`.
- Same-class replacement ở vùng đã phát hiện không được tính là novel detection.

Inference mặc định chạy crop theo batch nhỏ (`crop_batch_size: 3`) và dừng sau `max_consecutive_rejections: 2`. Có thể dùng full batched mode bằng `batched_inference: true`, nhưng mode này không tiết kiệm crop nhờ early-stop.

Metadata inference ghi accepted/rejected ROI, rejection reason, novel count, utility, max novel score, global stop reason và timing theo stage.

## Benchmark

Smoke benchmark:

```powershell
python scripts\benchmark.py --split test --limit 100 --out-dir runs\benchmark\smoke_100
```

Full benchmark:

```powershell
python scripts\benchmark.py --split test --out-dir runs\benchmark\full_test
```

Thesis benchmark (all 10 VisDrone classes, YOLO + slice budgets 4/6/12/15 + RL-SAHI):

```powershell
python scripts\benchmark.py --config configs\benchmark_thesis.yaml --split test --out-dir runs\benchmark\thesis_full_test
python scripts\benchmark.py --config configs\benchmark_thesis.yaml --split val --out-dir runs\benchmark\thesis_full_val
python scripts\build_benchmark_tables.py `
  --test-json runs\benchmark\thesis_full_test\benchmark.json `
  --val-json runs\benchmark\thesis_full_val\benchmark.json `
  --output runs\benchmark\thesis\benchmark_tables.md
python scripts/run_component_ablation.py --config configs/benchmark_component_ablation.yaml --split val --policy-device cuda    
```

`benchmark_thesis.yaml` follows the thesis class scope instead of the six-class training default.
It disables unrelated proposal/gated variants so the full run measures only the methods used in
the thesis tables. Output additionally includes Precision, Recall, images/s, detector calls and
Effective GFLOPs.

Benchmark hiện so sánh:

- `yolo_full`
- fixed-grid SAHI
- fixed-grid budget 4/8/12
- proposal SAHI
- gated variants dùng đúng acceptance gate của RL-SAHI
- `rl_sahi`

Output bao gồm AP kiểu IoU `0.50:0.95`, AP50, AP75, per-class AP/AP50/FP, small recall, FP/image, processed/accepted crops, acceptance rate, incremental latency, initial-state latency và end-to-end latency.

`benchmark.json` lưu full inference/benchmark config, image list, device, Git revision, weights/checkpoint SHA-256 và cache mode. `small_area_ratio` mặc định cố định ở `0.0004`, nên small recall có thể so sánh giữa các run có số ảnh khác nhau.

## Kiểm thử

Project dùng standard-library `unittest`:

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q src scripts tests
```

## Cấu trúc

```text
configs/                 Cấu hình path, detection, RL và inference
scripts/
  detect.py              Build detection/state cache
  hard_region.py         Build hard-region cache
  train.py               Train batched Dueling Double DQN
  infer.py               Adaptive slicing inference
  visualize.py           Render visualization từ metadata
  benchmark.py           Benchmark full/fixed/proposal/gated/RL
src/rl_sahi/
  common/                Config, cache, class, box và device helpers
  detection/             YOLO wrapper và feature extraction
  hard_region/           Hard-target mining
  inference/             ROI rollout, crop inference, acceptance và merge
  rl/                    Environment, state, network, replay và trainer
  eval/                  Metrics, sampling và benchmark manifest
tests/                   Unit tests và cache checker
```

## Quy tắc cập nhật artifact

Sau khi thay weights, class scope, detection state, hard-region mining, acceptance gate hoặc reward:

1. Rebuild cache liên quan.
2. Train mới nếu state/reward/class scope thay đổi.
3. Chạy unit tests và cache checker.
4. Benchmark stratified/full test.
5. Chỉ cập nhật bảng kết quả sau khi manifest chứng minh weights, checkpoint, config và image list khớp.
