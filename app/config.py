from pydantic import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    MODEL_PATH: str = "models/best_model.pth"
    DEVICE: str = "cpu"
    SCORE_THRESH: float = 0.25
    NUM_CLASSES: int = 7  # Adjust based on your training
    NUM_THREADS: int = 1

    class Config:
        env_file = ".env"

settings = Settings()
Path('models').mkdir(exist_ok=True)