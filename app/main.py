import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.model import get_model
from app.schemas import PredictRequest, PredictResponse
from app.utils import preprocess

# --- APP ---
app = FastAPI(title="Letter Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LABELS ---
label_map = {i: chr(64 + i) for i in range(1, 27)}

# --- MODEL LOAD (ОДИН РАЗ) ---
device = torch.device("cpu")
model = get_model()
model.to(device)
model.eval()


# --- HEALTH ---
@app.get("/health")
def health():
    return {"status": "ok"}


# --- FRONTEND ---
@app.get("/")
def root():
    return FileResponse("frontend/index.html")


# --- PREDICT ---
@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    try:
        arr = preprocess(data.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    try:
        x = torch.tensor(arr, dtype=torch.float32)
        x = x.view(1, 1, 28, 28)  # 🔥 важно для CNN
        x = x / 255.0             # 🔥 нормализация как при обучении
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tensor error: {str(e)}")

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    top5_idx = np.argsort(probs)[-5:][::-1]

    return PredictResponse(
        prediction=label_map[pred_idx + 1],
        top5=[
            {"letter": label_map[i + 1], "prob": float(probs[i])}
            for i in top5_idx
        ],
    )