import os
import logging
from ultralytics import YOLO

# Cấu hình Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PCBYOLOPredictor:
    """
    Module Suy luận Đa năng:
    - get_raw_predictions(): Trả về chuỗi JSON (Dùng cho Web API)
    - predict_and_save(): Vẽ ảnh và lưu lại (Dùng cho Local Test)
    """

    def __init__(self, weights_path: str = "best.pt"):
        self.weights_path = weights_path
        self.model = None
        self.load_model()  # Tự động load model ngay khi khởi tạo Class

    def load_model(self):
        if not os.path.exists(self.weights_path):
            logging.error(f"Không tìm thấy file trọng số tại: {self.weights_path}")
            return False

        logging.info(f"Đang tải mô hình từ {self.weights_path}...")
        self.model = YOLO(self.weights_path)
        return True

    def get_raw_predictions(self, image_input, conf_threshold: float = 0.5):
        """Hàm Lõi: Nhận vào đường dẫn file hoặc đối tượng ảnh (PIL/Numpy), trả về Dictionary"""
        if not self.model:
            return {"error": "Model chưa được nạp!"}

        # Chạy dự đoán (Chạy ngầm, không bật popup show/save để tối ưu RAM cho Server)
        results = self.model.predict(source=image_input, conf=conf_threshold)

        detections = []
        for r in results:
            for box in r.boxes:
                # Trích xuất tọa độ và thông tin
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names[cls_id]

                detections.append({
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                })

        return {"total_defects": len(detections), "detections": detections}

    def predict_and_save(self, source_path: str, conf_threshold: float = 0.5):
        """Hàm Phụ trợ: Vẽ khung lên ảnh và lưu ổ cứng (Cho Kỹ sư test Local)"""
        if not self.model:
            return None

        logging.info(f"Đang soi lỗi trên ảnh: {source_path}")
        results = self.model.predict(
            source=source_path,
            conf=conf_threshold,
            show=True,  # Bật popup
            save=True,  # Lưu ảnh
            line_width=2,
            name="my_results",
            exist_ok=True
        )
        return results


if __name__ == "__main__":
    # Test thử file này dưới Local
    predictor = PCBYOLOPredictor("../model/best.pt")
    test_img = "YOLO_DATASET/images/test/01_missing_hole_04.jpg"

    # 1. Test xem nó in ra JSON có chuẩn không
    print("--- TEST HÀM JSON ---")
    json_data = predictor.get_raw_predictions(test_img, conf_threshold=0.5)
    print(json_data)

    # 2. Test xem nó có vẽ ảnh không
    print("\n--- TEST HÀM VẼ ẢNH ---")
    predictor.predict_and_save(test_img, conf_threshold=0.5)