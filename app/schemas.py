from pydantic import BaseModel
from typing import List, Dict


class ArrayRequest(BaseModel):
    pixels: List[float]


class PredictResponse(BaseModel):
    predicted_letter: str
    predicted_index: int
    confidence: float
    probabilities: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    device: str
    classes: List[str]
    model_path: str