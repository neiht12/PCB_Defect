import glob
import shutil
import random
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple

# 1. Cấu hình Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class VOC2YOLOConverter:
    """
    Module tiền xử lý dữ liệu: Chuyển đổi định dạng PASCAL VOC sang YOLO.
    """

    def __init__(self, root_dir: str, output_dir: str, classes: Dict[str, int],
                 split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)):

        self.root_dir = Path(root_dir)
        self.xml_dir = self.root_dir / "Annotations"
        self.img_dir = self.root_dir / "images"
        self.output_dir = Path(output_dir)

        self.classes = classes
        self.split_ratio = split_ratio

        self.dirs = {
            'train_img': self.output_dir / 'images' / 'train',
            'val_img': self.output_dir / 'images' / 'val',
            'test_img': self.output_dir / 'images' / 'test',
            'train_lbl': self.output_dir / 'labels' / 'train',
            'val_lbl': self.output_dir / 'labels' / 'val',
            'test_lbl': self.output_dir / 'labels' / 'test'
        }

    def setup_directories(self):
        """Tạo cây thư mục YOLO."""
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        logging.info("Đã thiết lập xong cấu trúc thư mục YOLO.")

    def convert_bbox(self, img_size: Tuple[int, int], box: Tuple[float, float, float, float]) -> Tuple[
        float, float, float, float]:
        """Chuẩn hóa tọa độ bounding box."""
        dw = 1.0 / img_size[0]
        dh = 1.0 / img_size[1]
        x_center = (box[0] + box[1]) / 2.0
        y_center = (box[2] + box[3]) / 2.0
        width = box[1] - box[0]
        height = box[3] - box[2]

        return (x_center * dw, y_center * dh, width * dw, height * dh)

    def process_split(self, xml_files: List[str], img_dest: Path, lbl_dest: Path):
        """Xử lý một tập dữ liệu (Train/Val/Test)."""
        success_count = 0

        for xml_path_str in xml_files:
            try:
                xml_path = Path(xml_path_str)
                # Thay thế chuỗi để tìm đường dẫn ảnh tương ứng
                img_path = Path(str(xml_path).replace("Annotations", "images").replace(".xml", ".jpg"))

                # Kiểm tra ảnh có tồn tại không
                if not img_path.exists():
                    logging.warning(f"Bỏ qua: Không tìm thấy ảnh {img_path}")
                    continue

                tree = ET.parse(xml_path)
                root = tree.getroot()

                img_width = int(root.find("size/width").text)
                img_height = int(root.find("size/height").text)

                txt_filename = img_path.with_suffix('.txt').name
                txt_filepath = lbl_dest / txt_filename

                # Ghi file nhãn
                with open(txt_filepath, "w") as o_file:
                    for obj in root.iter("object"):
                        class_name = obj.find("name").text
                        if class_name in self.classes:
                            class_id = self.classes[class_name]

                            b = (
                                float(obj.find("bndbox/xmin").text),
                                float(obj.find("bndbox/xmax").text),
                                float(obj.find("bndbox/ymin").text),
                                float(obj.find("bndbox/ymax").text)
                            )
                            bb = self.convert_bbox((img_width, img_height), b)

                            o_file.write(f"{class_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

                # Copy ảnh
                shutil.copy(img_path, img_dest / img_path.name)
                success_count += 1

            except Exception as e:
                logging.error(f"Lỗi khi xử lý file {xml_path_str}: {e}")

        return success_count

    def run_pipeline(self):
        """Khởi chạy toàn bộ quy trình tiền xử lý."""
        logging.info("BẮT ĐẦU DATA PREPROCESSING PIPELINE...")
        self.setup_directories()

        # Thu thập và xáo trộn dữ liệu
        xml_data_files = glob.glob(str(self.xml_dir / "**" / "*.xml"), recursive=True)
        random.shuffle(xml_data_files)
        total = len(xml_data_files)

        if total == 0:
            logging.error("Không tìm thấy file XML nào. Vui lòng kiểm tra lại đường dẫn!")
            return

        # Tính toán chia tỷ lệ
        train_ratio, val_ratio, _ = self.split_ratio
        train_end = int(train_ratio * total)
        val_end = int((train_ratio + val_ratio) * total)

        train_files = xml_data_files[:train_end]
        val_files = xml_data_files[train_end:val_end]
        test_files = xml_data_files[val_end:]

        # Xử lý từng tập
        logging.info(f"Đang xử lý tập Train ({len(train_files)} files)...")
        train_success = self.process_split(train_files, self.dirs['train_img'], self.dirs['train_lbl'])

        logging.info(f"Đang xử lý tập Val ({len(val_files)} files)...")
        val_success = self.process_split(val_files, self.dirs['val_img'], self.dirs['val_lbl'])

        logging.info(f"Đang xử lý tập Test ({len(test_files)} files)...")
        test_success = self.process_split(test_files, self.dirs['test_img'], self.dirs['test_lbl'])

        logging.info(f"HOÀN TẤT! Thành công: Train({train_success}), Val({val_success}), Test({test_success})")


# ==========================================
# KHỐI THỰC THI CHÍNH (ENTRY POINT)
# ==========================================
if __name__ == "__main__":
    # Định nghĩa cấu hình
    CLASSES = {
        "missing_hole": 0,
        "mouse_bite": 1,
        "open_circuit": 2,
        "short": 3,
        "spur": 4,
        "spurious_copper": 5
    }

    # Khởi tạo và chạy Pipeline
    pipeline = VOC2YOLOConverter(
        root_dir="PCB_DATASET",
        output_dir="YOLO_DATASET",
        classes=CLASSES,
        split_ratio=(0.8, 0.1, 0.1)
    )

    pipeline.run_pipeline()