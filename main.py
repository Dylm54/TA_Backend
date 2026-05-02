from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import os
import uuid
import cv2
import base64
import time
import psutil

app = FastAPI()

# CORS (DEV MODE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD MODEL
model = YOLO("model/best.pt")

# RULE SEVERITY
def map_severity(count):
    if count <= 5:
        return "Mild"
    elif count <= 20:
        return "Moderate"
    elif count <= 50:
        return "Severe"
    else:
        return "Very Severe"

# ENDPOINT
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    t0 = time.perf_counter()

    file_path = f"temp_{uuid.uuid4()}.jpg"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    t1 = time.perf_counter()  # file save selesai

    try:
        # INFERENCE
        results = model(file_path, conf=0.15)

        t2 = time.perf_counter()  # inference selesai

        boxes = results[0].boxes
        count = 0 if boxes is None else len(boxes)
        severity = map_severity(count)

        # LOAD IMAGE
        img = cv2.imread(file_path)

        t3 = time.perf_counter()  # load image selesai

        # DRAW
        if boxes is not None:
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(img, (x1, y1), (x2, y2), (245, 85, 97), 2)

        t4 = time.perf_counter()  # drawing selesai

        # ENCODE
        _, buffer = cv2.imencode(".jpg", img)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        t5 = time.perf_counter()  # encode selesai

        # LOG SEMUA
        print("==== PERFORMANCE ====")
        print(f"Save file     : {(t1 - t0):.3f}s")
        print(f"Inference     : {(t2 - t1):.3f}s")
        print(f"Load image    : {(t3 - t2):.3f}s")
        print(f"Draw boxes    : {(t4 - t3):.3f}s")
        print(f"Encode base64 : {(t5 - t4):.3f}s")
        print(f"TOTAL         : {(t5 - t0):.3f}s")

        # RAM
        process = psutil.Process(os.getpid())
        print("Memory (MB):", process.memory_info().rss / 1024 / 1024)

        return {
            "lesion_count": count,
            "severity": severity,
            "image": encoded_image
        }

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)