import logging
from core.data_preprocess import VOC2YOLOConverter
from core.train import PCBYOLOTrainer
from core.inference import PCBYOLOPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_ml_pipeline():
    logging.info(" PCB DEFECT DETECTION PIPELINE...")

    RUN_DATA_PREP = False  # Bật True nếu có dữ liệu XML mới cần convert
    RUN_TRAINING = False  # Bật True nếu muốn train lại mô hình
    RUN_INFERENCE = True  # Bật True để test mô hình với ảnh

    if RUN_DATA_PREP:
        logging.info(">>> STEP 1: PREPROCESSING DATA <<<")
        CLASSES = {
            "missing_hole": 0, "mouse_bite": 1, "open_circuit": 2,
            "short": 3, "spur": 4, "spurious_copper": 5
        }
        converter = VOC2YOLOConverter(
            root_dir="PCB_DATASET",
            output_dir="YOLO_DATASET",
            classes=CLASSES
        )
        converter.run_pipeline()

    if RUN_TRAINING:
        logging.info(">>> STEP 2: TRAINING <<<")
        trainer = PCBYOLOTrainer(data_yaml="dataset.yaml", model_type="yolov8n.pt")
        trainer.train(epochs=50, image_size=640, batch_size=16)

    if RUN_INFERENCE:
        logging.info(">>> STEP 3: INFERENCE <<<")
        predictor = PCBYOLOPredictor(weights_path="model/best.pt")
        test_image = "YOLO_DATASET/images/test/01_missing_hole_04.jpg"
        predictor.predict_image(source_path=test_image, conf_threshold=0.5)

    logging.info("PIPELINE HAS COMPLETED!")


if __name__ == "__main__":
    run_ml_pipeline()