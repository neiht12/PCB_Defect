import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from core.inference import PCBYOLOPredictor


# Khởi tạo app (Giữ nguyên dòng này của bạn)
app = FastAPI(title="PCB Defect Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép mọi trang web gọi API này
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Đang khởi động Server và nạp AI...")
predictor = PCBYOLOPredictor("model/best.pt")

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result_dict = predictor.get_raw_predictions(image, conf_threshold=0.5)
        result_dict["filename"] = file.filename

        return JSONResponse(content=result_dict)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
def health_check():
    return {"status": "Server đang chạy rất mượt!", "model": "YOLOv8 Nano"}