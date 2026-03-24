import io
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch.nn.functional as F
import os

from app.schemas import PredictResponse, ArrayRequest, HealthResponse
from app.model import load_model, get_device

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "model/letter_cnn.pth")
device = get_device()
model = load_model(MODEL_PATH, device)

LABELS = [chr(ord('A') + i) for i in range(26)]


# ─────────────────────────────────────────────
# Базовый инференс
# ─────────────────────────────────────────────
def infer_tensor(tensor: torch.Tensor):
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


# ─────────────────────────────────────────────
# Rotation + best selection
# ─────────────────────────────────────────────
def predict_with_rotations(img: Image.Image) -> PredictResponse:
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    weights = {
        0: 1.0,
        45: 0.8,
        90: 0.3,
        135: 0.2,
        180: 0.1,
        225: 0.2,
        270: 0.3,
        315: 0.8,
    }

    best_probs = None
    best_score = -1

    for angle in angles:
        rotated = img.rotate(angle, fillcolor=255)

        X = 1.0 - (np.array(rotated, dtype=np.float32) / 255.0)
        tensor = torch.tensor(X).reshape(1, 1, 28, 28)

        probs = infer_tensor(tensor)

        score = probs.max() * weights[angle]

        if score > best_score:
            best_score = score
            best_probs = probs

    idx = int(best_probs.argmax())
    confidence = float(best_probs[idx])

    result = PredictResponse(
        predicted_letter=LABELS[idx],
        predicted_index=idx,
        confidence=confidence,
        probabilities={l: float(p) for l, p in zip(LABELS, best_probs)},
    )

    # защита от переуверенности
    if confidence < 0.6:
        result.predicted_letter = "?"

    return result


# ─────────────────────────────────────────────
# API
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        device=str(device),
        classes=LABELS,
        model_path=MODEL_PATH,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_array(body: ArrayRequest):
    if len(body.pixels) != 784:
        raise HTTPException(400, "Нужно 784 пикселя")

    X = np.array(body.pixels, dtype=np.float32) / 255.0
    tensor = torch.tensor(X).reshape(1, 1, 28, 28)

    probs = infer_tensor(tensor)

    idx = int(probs.argmax())

    return PredictResponse(
        predicted_letter=LABELS[idx],
        predicted_index=idx,
        confidence=float(probs[idx]),
        probabilities={l: float(p) for l, p in zip(LABELS, probs)},
    )


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents)).convert("L")
        img = img.resize((28, 28))
    except Exception as e:
        raise HTTPException(400, str(e))

    return predict_with_rotations(img)


# ─────────────────────────────────────────────
# Frontend
# ─────────────────────────────────────────────
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")