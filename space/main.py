"""
MediScan AI — HuggingFace Space Backend
Port 7860 (required by HuggingFace Spaces)
"""

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
import io

from model_loader import (
    load_pneumo_model, load_skin_models,
    predict_pneumonia, predict_skin
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 50)
    print("  MediScan AI Space — Loading models...")
    print("=" * 50)
    load_pneumo_model()
    load_skin_models()
    print("=" * 50)
    print("  All models ready!")
    print("=" * 50)
    yield


app = FastAPI(
    title="MediScan AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "endpoints": {
            "pneumonia": "POST /predict/pneumonia",
            "skin":      "POST /predict/skin",
            "docs":      "/docs",
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict/pneumonia")
async def pneumonia_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Must be an image file.")
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Could not read image.")
    try:
        return predict_pneumonia(image)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")


@app.post("/predict/skin")
async def skin_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Must be an image file.")
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Could not read image.")
    try:
        return predict_skin(image)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
