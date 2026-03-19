from ultralytics import YOLO
import logging

class PCBYOLOTrainer:
    def __init__(self, data_yaml: str, model_type : str = "yolov8n.pt"):
        self.data_yaml = data_yaml
        self.model_type = model_type
        self.model = None
    def load_model(self):
        logging.info(f"Loading Model {self.model_type} ")
        self.model = YOLO(self.model_type)

    def train(self, epochs: int = 50, image_size: int = 640, batch_size: int = 16):
        if not self.model:
            self.load_model()

        logging.info("TRAINING PHASE (TRAINING)...")
        logging.info(f" Epochs={epochs}, Image Size={image_size}, Batch Size={batch_size}")
        results = self.model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=image_size,
            batch=batch_size,
            project="PCB_Defect_Detection",
            name="yolov8n_training",
            patience=10,
            device = "cuda"
        )
        logging.info("TRAINING SUCCESSFUL")
        return results

    def export_to_onnx(self):
        logging.info("Convert to ONNX...")
        if self.model:
            success = self.model.export(format='onnx', opset=12)
            logging.info(f"Convert complete.")
        else:
            logging.error("Dont have model to convert. Loadding model.")


if __name__ == "__main__":
    # Khai báo đường dẫn tới file yaml vừa tạo
    YAML_PATH = "dataset.yaml"

    # Sử dụng bản Nano (yolov8n.pt) siêu nhẹ, chạy cực nhanh
    trainer = PCBYOLOTrainer(data_yaml=YAML_PATH, model_type="yolov8n.pt")

    # Bắt đầu train. Nếu máy yếu, bạn có thể giảm batch xuống 8.
    trainer.train(epochs=50, image_size=640, batch_size=16)

    # (Tùy chọn) Mở comment dòng dưới đây để tự động xuất file ONNX sau khi train xong
    # trainer.export_to_onnx()