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


# ── Inference ────────────────────────────────────────────────────────────────
def run_inference(tensor: torch.Tensor) -> PredictResponse:
    with torch.no_grad():
        logits = model(tensor.to(device))

        # ↓ уменьшаем "самоуверенность"
        probs = F.softmax(logits / 2.0, dim=1).cpu().numpy()[0]

        idx = int(probs.argmax())
        confidence = float(probs[idx])

    return PredictResponse(
        predicted_letter=LABELS[idx],
        predicted_index=idx,
        confidence=confidence,
        probabilities={l: float(p) for l, p in zip(LABELS, probs)},
    )


# ── Rotation ensemble ────────────────────────────────────────────────────────
def predict_with_rotations(img: Image.Image) -> PredictResponse:
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    best_result = None
    best_conf = -1

    for angle in angles:
        rotated = img.rotate(angle, fillcolor=255)

        X = 1.0 - (np.array(rotated, dtype=np.float32) / 255.0)
        tensor = torch.tensor(X).reshape(1, 1, 28, 28)

        result = run_inference(tensor)

        if result.confidence > best_conf:
            best_conf = result.confidence
            best_result = result

    # threshold (если не уверен)
    if best_result.confidence < 0.85:
        best_result.predicted_letter = "?"

    return best_result


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        device=str(device),
        classes=LABELS,
        model_path=MODEL_PATH,
    )


# ── Predict array ────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse)
def predict_array(body: ArrayRequest):
    if len(body.pixels) != 784:
        raise HTTPException(400, "Нужно 784 пикселя")

    X = np.array(body.pixels, dtype=np.float32) / 255.0
    tensor = torch.tensor(X).reshape(1, 1, 28, 28)
    return run_inference(tensor)


# ── Predict image ────────────────────────────────────────────────────────────
@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents)).convert("L")
        img = img.resize((28, 28))
    except Exception as e:
        raise HTTPException(400, str(e))

    return predict_with_rotations(img)


# ── Frontend ─────────────────────────────────────────────────────────────────
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")