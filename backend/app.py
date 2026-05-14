# backend/app.py - Optimized for Render deployment
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import uuid
import os
import time

# Initialize FastAPI
app = FastAPI(
    title="Traffic Sign Detection API",
    description="Real-time traffic sign detection using YOLOv8",
    version="1.0.0"
)

# CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Class labels (based on your mapping)
CLASS_LABELS = {
    0: "Information", 7: "20 km/h", 8: "30 km/h", 9: "40 km/h",
    10: "60 km/h", 11: "50 km/h", 12: "60 km/h", 16: "No Vehicles",
    22: "Pedestrian", 23: "Traffic Sign", 24: "Ahead Only",
    25: "Parking", 26: "Children Crossing", 31: "Two-way Traffic",
    33: "Traffic Sign", 34: "Roundabout", 35: "Straight/Right",
    36: "Straight Only", 37: "Straight Only", 38: "Yield",
    39: "Priority Road", 40: "STOP", 41: "Children Crossing",
    42: "Road Work",
}


def get_label(class_id):
    return CLASS_LABELS.get(class_id, f"Sign_{class_id}")


def load_model():
    """Load YOLO model"""
    global model
    try:
        # Try to load from model directory
        model_path = Path(__file__).parent.parent / 'model' / 'best.pt'
        if not model_path.exists():
            # Try alternative location
            model_path = Path('/app/model/best.pt')

        if model_path.exists():
            model = YOLO(str(model_path))
            print(f"✅ Model loaded from {model_path}")
        else:
            # Fallback to default model
            print("⚠️ No trained model found, using default YOLOv8")
            model = YOLO('yolov8n.pt')

        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()
    print(f"🚀 API Started on {device}")
    print(f"📊 Model ready: {model is not None}")


@app.get("/")
async def root():
    return {
        "name": "Traffic Sign Detection API",
        "status": "running",
        "device": device,
        "model_loaded": model is not None,
        "endpoints": ["/health", "/detect", "/info", "/detect/url"]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }


@app.get("/info")
async def get_info():
    return {
        "model": "YOLOv8",
        "device": device,
        "classes": len(CLASS_LABELS),
        "class_mapping": CLASS_LABELS
    }


@app.post("/detect")
async def detect_signs(file: UploadFile = File(...), confidence: float = 0.25):
    """Detect traffic signs in uploaded image"""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Run detection
    results = model(image, conf=confidence)

    detections = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls,
                "label": get_label(cls)
            })

    processing_time = time.time() - start_time

    return {
        "success": True,
        "detections": detections,
        "count": len(detections),
        "processing_time": processing_time,
        "image_size": {"width": image.shape[1], "height": image.shape[0]}
    }


@app.post("/detect/batch")
async def detect_batch(files: List[UploadFile] = File(...), confidence: float = 0.25):
    """Detect signs in multiple images"""

    results = []
    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is not None:
                detections = model(image, conf=confidence)
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "count": len(detections[0].boxes) if detections[0].boxes else 0
                })
            else:
                results.append({"filename": file.filename, "success": False})
        except Exception as e:
            results.append({"filename": file.filename, "success": False, "error": str(e)})

    return {"results": results, "total": len(files)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)