from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    MODEL_PATH: str = "models/best_model.pth"
    DEVICE: str = "cpu"          # "cuda" if you have GPU in prod
    SCORE_THRESH: float = 0.25
    NUM_CLASSES: int = 7         # <- match your training
    NUM_THREADS: int = 1

    class Config:
        env_file = ".env"

settings = Settings()
Path("models").mkdir(exist_ok=True)