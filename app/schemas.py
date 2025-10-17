from pydantic import BaseModel
from typing import List, Optional

class Box(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    score: float
    label_id: int
    label: Optional[str] = None

class PredictImageResponse(BaseModel):
    detections: List[Box]
    num_detections: int