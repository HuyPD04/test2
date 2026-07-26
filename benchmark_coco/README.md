# VisDrone to COCO Benchmark Module (`benchmark_coco`)

Module này được xây dựng độc lập trong thư mục `D:\RL-SAHI\benchmark_coco` để thực hiện chuyển đổi dữ liệu annotation của VisDrone và kết quả dự đoán (inference predictions) sang chuỗi định dạng tiêu chuẩn **MS COCO JSON**, sau đó đánh giá độ chính xác thông qua `pycocotools.cocoeval.COCOeval`.

---

## 1. Cấu trúc hoạt động

Script chính: [`eval_coco.py`](file:///D:/RL-SAHI/benchmark_coco/eval_coco.py) thực hiện quy trình 3 bước tự động:
1. **Chuyển đổi Ground Truth (GT)**: 
   - Đọc các file `.txt` trong `D:\RL-SAHI\data\raw\VisDrone2019-DET-test-dev\annotations`.
   - Tự động lấy kích thước ảnh tương ứng trong folder `images/` (để tính toán chuẩn xác tỷ lệ).
   - Loại bỏ các vùng ignore (score=0, category=0 hoặc >10).
   - Lưu kết quả chuẩn COCO vào file `visdrone_test_dev_gt.json`.
2. **Chuyển đổi Predictions (DT)**:
   - Đọc các file dự đoán `.txt` trong `D:\RL-SAHI\runs\infer\detections` (hỗ trợ cả định dạng YOLO `class score x1 y1 x2 y2` lẫn định dạng chính thức VisDrone `x1, y1, w, h, score, class`).
   - Ghép nối ID ảnh với GT theo tên file (`stem`).
   - Chuyển Bounding Box về định dạng `[x_min, y_min, width, height]` và map class ID về 1..10.
   - Lưu kết quả vào file `predictions_coco.json`.
3. **Đánh giá COCOeval**:
   - Khởi tạo đối tượng `COCO` và `COCOeval(iouType='bbox')`.
   - Tính toán 12 chỉ số tổng thể theo chuẩn MS COCO (AP 0.50:0.95, AP50, AP75, AP theo kích thước small/medium/large, và Recall AR).
   - Tính toán chi tiết mAP và AP50 cho từng class (Pedestrian, Car, Bus, Motor,...).
   - Xuất bảng tóm tắt ra terminal, đồng thời lưu báo cáo đầy đủ vào `eval_results.json` và `eval_summary.txt`.

---

## 2. Hướng dẫn sử dụng

Vì thư viện `pycocotools` đã có sẵn trong môi trường conda **`doan`** của bạn, chỉ cần mở terminal và chạy lệnh:

```bash
# Kích hoạt môi trường conda
conda activate doan

# Di chuyển vào folder gốc của project
cd /d D:\RL-SAHI

# Chạy script đánh giá với cấu hình mặc định
python benchmark_coco/eval_coco.py
```

### Tuỳ biến tham số (tùy chọn)
Nếu bạn muốn đánh giá trên tập dữ liệu khác hoặc folder detections khác, có thể dùng các cờ CLI:

```bash
python benchmark_coco/eval_coco.py \
  --gt-dir "D:/RL-SAHI/data/raw/VisDrone2019-DET-test-dev/annotations" \
  --img-dir "D:/RL-SAHI/data/raw/VisDrone2019-DET-test-dev/images" \
  --dt-dir "D:/RL-SAHI/runs/infer/detections" \
  --out-dir "D:/RL-SAHI/benchmark_coco"
```

- `--skip-eval`: Chỉ chuyển đổi ra 2 file JSON (`gt.json` và `dt.json`) mà không chạy COCOeval.

---

## 3. Các file kết quả đầu ra

Sau khi chạy thành công, folder `D:\RL-SAHI\benchmark_coco` sẽ chứa các file:
- `visdrone_test_dev_gt.json`: File Ground Truth chuẩn COCO JSON.
- `predictions_coco.json`: File Predictions chuẩn COCO JSON.
- `eval_results.json`: Báo cáo chỉ số chi tiết dưới dạng JSON (dùng để vẽ biểu đồ hoặc phân tích chuyên sâu).
- `eval_summary.txt`: Bảng tóm tắt kết quả format đẹp mắt.
