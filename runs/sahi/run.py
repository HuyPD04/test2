import time
import cv2
import numpy as np
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def benchmark_sahi_folder(folder_path: str, model_path: str, output_dir: str, ext: str = "*.jpg"):
    image_paths = list(Path(folder_path).glob(ext))
    num_images = len(image_paths)
    
    if num_images == 0:
        print(f"Không tìm thấy ảnh định dạng {ext} trong thư mục {folder_path}")
        return

    # Tạo thư mục lưu kết quả nếu chưa có
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Kết quả sẽ được lưu tại: {out_path.resolve()}")

    print(f"Tìm thấy {num_images} ảnh. Bắt đầu tải mô hình...")

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",          
        model_path=model_path,
        confidence_threshold=0.3,
        device="cuda:0"               
    )

    print("Đang làm nóng hệ thống (Warm-up)...")
    dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    get_sliced_prediction(
        dummy_image, detection_model, slice_height=512, slice_width=512
    )

    total_time = 0.0
    print(f"Bắt đầu chạy suy luận (Inference) trên {num_images} ảnh...")

    for i, img_path in enumerate(image_paths, 1):
        start_time = time.perf_counter()
        
        # Nhận đối tượng kết quả trả về thay vì dùng '_'
        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            verbose=0 
        )

        end_time = time.perf_counter()
        
        iter_time = end_time - start_time
        total_time += iter_time
        
        # --- BẮT ĐẦU PHẦN GHI FILE .TXT ---
        # Đặt tên file .txt giống hệt tên file ảnh
        txt_filename = out_path / f"{img_path.stem}.txt"
        
        img_w = result.image_width
        img_h = result.image_height
        
        with open(txt_filename, "w", encoding="utf-8") as f:
            for obj in result.object_prediction_list:
                cls_id = obj.category.id
                score = obj.score.value
                
                # SAHI trả về bounding box dạng [xmin, ymin, width, height]
                x_min, y_min, w, h = obj.bbox.to_xywh()
                
                # Chuyển đổi sang định dạng YOLO chuẩn (chuẩn hóa từ 0-1)
                x_center = (x_min + w / 2) / img_w
                y_center = (y_min + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # Ghi ra file với định dạng: class_id score x_center y_center width height
                # (Đối với dự đoán, các tool benchmark thường yêu cầu thêm cột score)
                f.write(f"{cls_id} {score:.6f} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        # --- KẾT THÚC PHẦN GHI FILE .TXT ---

        if i % 10 == 0 or i == num_images:
            print(f"Đã xử lý {i}/{num_images} ảnh...")

    avg_time = total_time / num_images
    fps = 1.0 / avg_time

    print("-" * 30)
    print("KẾT QUẢ BENCHMARK SAHI (512x512)")
    print(f"Tổng số ảnh     : {num_images}")
    print(f"Tổng thời gian  : {total_time:.2f} giây")
    print(f"Trung bình 1 ảnh: {avg_time:.4f} giây/ảnh")
    print(f"Tốc độ khung hình: {fps:.2f} FPS")
    print("-" * 30)

if __name__ == "__main__":
    FOLDER_PATH = r"data\raw\images\test"
    MODEL_PATH = "yolo11l.pt"  
    OUTPUT_DIR = r"benchmark_official\results\test"

    benchmark_sahi_folder(FOLDER_PATH, MODEL_PATH, OUTPUT_DIR, ext="*.jpg")