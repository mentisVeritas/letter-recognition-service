from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    image: str  # base64

class TopItem(BaseModel):
    letter: str
    prob: float

class PredictResponse(BaseModel):
    prediction: str
    top5: List[TopItem]