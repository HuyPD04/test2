# Anchor-Zoom RL

Đây là bản cài đặt độc lập của phương pháp A: RL ra quyết định tuần tự ở cấp
crop. Mỗi bước chọn trực tiếp một cặp `anchor_id x zoom_bin` hoặc `STOP`; không
còn rollout 11 action di chuyển ROI.

Code trong thư mục này không import `src/rl_sahi` và không cần script/config ở
cấp repo. Dataset và weights chỉ được truyền vào bằng đường dẫn trong
`configs/default.yaml`, vì vậy có thể chép nguyên thư mục sang một project khác.

## Luồng phương pháp

```text
ảnh
  -> YOLO full-image một lần
  -> gom detection theo ô lưới thành top-K anchor
  -> state = anchor features + history + crop budget
  -> Dueling Double DQN chọn anchor x zoom hoặc STOP
  -> crop YOLO
  -> merge class-aware NMS
  -> reward TP/hard-TP gain + utility - FP/crop/overlap cost
  -> cập nhật history và chọn crop tiếp theo
```

Một anchor chỉ được chọn một lần. Action tạo ROI trùng history quá ngưỡng bị
mask. Cùng một implementation của anchor, state, action mask, crop acceptance và
merge được dùng trong cả train lẫn infer.

## Cấu trúc

```text
anchor_zoom_rl/
  anchor_zoom_rl/
    core/       anchor, hard region, geometry, state, NMS, reward
    rl/         Dueling DQN, Double-DQN update, replay, n-step return
    runtime/    data, YOLO, cache, environment, train, infer, metrics
  configs/      cấu hình độc lập
  scripts/      train, infer, evaluate, precompute
  tests/        unit test và infer smoke test không cần YOLO
  runs/         checkpoint, log và kết quả
```

## Cài đặt

Chạy trong `method/anchor_zoom_rl`:

```powershell
python -m pip install -r requirements.txt
```

Mặc định config trỏ tới dataset và `yolo11s.pt` ở repo hiện tại. Khi chuyển
thư mục này đi nơi khác, chỉ cần sửa nhóm `paths`.

## Chạy

Precompute full-image detection và hard region trước khi train:

```powershell
python scripts/precompute.py --split train
python scripts/precompute.py --split val
```

Train:

```powershell
python scripts/train.py --episodes 20000 --out-dir runs/hard_aware_v3
```

Resume từ `runs/checkpoints/latest.pt`:

```powershell
python scripts/train.py --episodes 25000 --out-dir runs/hard_aware_v3 --resume
```

Train mặc định dùng `shuffled_epochs`: mọi ảnh được lấy đúng một lần trước khi
bắt đầu epoch kế tiếp. Với 6.471 ảnh train, 20.000 episode tương đương khoảng
3,1 epoch. Không resume checkpoint schema 2 sau khi đổi reward/network; hãy dùng
run directory mới. Cache hợp lệ được nhận diện bằng signature và có thể tái sử dụng.

Mỗi 1.000 episode, policy được đánh giá trên 512 ảnh val lấy stratified theo
sequence. `best.pt` là alias của `best_tradeoff.pt`; `best_ap.pt` và
`best_hard_recall.pt` được lưu độc lập. Toàn bộ lịch sử validation được lưu
trong `runs/eval.csv`.

Mỗi run phải dùng một `--out-dir` riêng. Fresh train sẽ dừng nếu thư mục đã có
`train.csv` hoặc `latest.pt`, tránh nối log/checkpoint của hai reward config.

Infer một ảnh hoặc cả split:

```powershell
python scripts/infer.py --image ..\..\data\raw\images\test\0000006_00159_d_0000001.jpg
python scripts/infer.py --split test
```

Kết quả detection được ghi thẳng theo format VisDrone chính thức:
`x,y,w,h,score,class_id_1_based,-1,-1`.

Đánh giá AP50 và latency:

```powershell
python scripts/evaluate.py --split val --limit 500
```

File JSON kết quả có ba trường latency:

- `initial_state_ms_per_image`: đọc ảnh + full-image YOLO hoặc cache lookup.
- `latency_ms_per_image`: phần còn lại sau initial state, gồm sinh anchor/state,
  policy, crop YOLO và quyết định accept/merge.
- `end_to_end_ms_per_image`: toàn bộ thời gian thực tế của pipeline.

Do đó `initial_state_ms_per_image + latency_ms_per_image` xấp xỉ
`end_to_end_ms_per_image`; sai số chỉ ở phép lấy trung bình/làm tròn.

Cache inference tắt mặc định để số latency phản ánh lần chạy thật. Chỉ bật
`inference.cache_full_detections` khi benchmark cố ý tách initial state đã
precompute.

Chạy test:

```powershell
python -m pytest
```

Nếu môi trường chỉ có runtime dependencies và chưa cài `pytest`:

```powershell
python scripts/smoke_test.py
```

## Training schema 3

Schema 3 changes the replay targets, reward, and network architecture. Start a fresh
run; do not resume a schema 2 checkpoint. Legacy checkpoints remain loadable for
inference through the unchanged Q head.

The hard-action auxiliary head predicts whether each valid `anchor x zoom` action
contains an unmatched hard object. Its target is available only during training
from the hard-region cache. Inference uses the learned shared representation and
does not require labels.

Validation writes hard recall, attempted/accepted hard coverage, and method
latency to `eval.csv`. Checkpoints are saved as:

```text
checkpoints/
  best.pt                 alias of best_tradeoff.pt
  best_tradeoff.pt        AP50 + hard recall - FP/crop cost
  best_ap.pt              highest AP50
  best_hard_recall.pt     highest hard-region recall
  latest.pt               resumable state with replay
```

## Cache

Dataset mặc định được đọc từ `D:\RL-SAHI\data\raw`. Tất cả cache nằm dưới
`D:\RL-SAHI\data\cache_1`:

```text
data/cache_1/
  detections/<split>/             full-image detection
  detections/crops/<split>/       crop detection sinh trong lúc train
  hard_regions/<split>/           hard mask, best IoU và best score theo GT
```

`precompute.py` sinh đồng thời `detections/<split>` và
`hard_regions/<split>`. Cache có chữ ký gồm ảnh, label, weights, kích thước
input, confidence, IoU và target classes; đổi dữ liệu hoặc cấu hình sẽ tự bỏ
qua cache cũ.

## Hard-region reward

Một GT được xem là hard khi full-image detector không match đúng class tại
`reward.match_iou`, hoặc match có score thấp hơn
`reward.hard_low_confidence`. Crop nhận bonus lớn `hard_tp_weight` khi detection
sau merge thực sự phục hồi hard GT. Lần đầu policy phủ một hard GT còn unmatched
nhận thêm shaping reward nhỏ, bị chặn bởi `hard_coverage_max_bonus`, để học đúng
vùng mà không thay thế mục tiêu detection.

Reward n-step đưa vào replay được clip ở `[-10, 10]`. Crop còn phải vượt
`reward.min_reliability`; reliability kết hợp cả detection mới và refinement
của detection cũ, anchor score, boundary ratio và overlap history. Sau
class-aware NMS, các box khác class gần như trùng nhau được loại bằng
`detector.cross_class_iou` và `detector.cross_class_ios`.

## Ghi chú đánh giá

`scripts/evaluate.py` cung cấp AP50 nội bộ để theo dõi nhanh AP-latency. Kết luận
cuối cùng trên VisDrone vẫn nên chạy bộ evaluator chính thức với các file trong
`runs/infer/predictions`.
