# PCB_Defect

# 🔍 PCB Defect Detection System (End-to-End AI Pipeline)

A professional-grade computer vision solution for automated Quality Control (QC) in PCB manufacturing. This project utilizes **YOLOv8** for real-time defect detection, wrapped in a **FastAPI** backend with a sleek **HTML5 Canvas** frontend.

---

## 🚀 Project Overview
This system identifies 6 common PCB defects:
* `missing_hole`
* `mouse_bite`
* `open_circuit`
* `short`
* `spur`
* `spurious_copper`

The project covers the entire ML lifecycle: **Data Transformation** -> **Model Training** -> **API Deployment** -> **Web Integration**.

## 📂 Project Structure
```text
PCB_DEFECT_PROJECT/
├── core/                # Core AI Logic (Modular)
│   ├── __init__.py
│   ├── data_prep.py     # XML to YOLO conversion
│   ├── train.py         # Training orchestration
│   └── inference.py     # Prediction engine (JSON & Visual)
├── models/              # Saved weights (best.pt)
├── data/                # Dataset storage
│   ├── PCB_DATASET/     # Raw images and XMLs
│   ├── YOLO_DATASET/    # Processed YOLO format data
│   └── dataset.yaml     # YOLO configuration file
├── frontend/            # Web Interface
│   └── index.html       # Frontend UI with Canvas rendering
├── app.py               # FastAPI Web Server (Production API)
├── pipeline.py          # Internal Orchestrator
└── requirements.txt     # Dependency list