import torch
from fastapi import FastAPI, UploadFile, File
from app.model import load_model
from app.predict_utils import preprocess_image, filter_predictions
from app.schemas import PredictImageResponse, Box
from app.config import settings

app = FastAPI(title="Faster R-CNN Deployment API")
model = None

@app.on_event("startup")
def startup_event():
    global model
    torch.set_num_threads(settings.NUM_THREADS)
    model = load_model()

@app.get("/health")
def health():
    return {"status": "ok", "device": settings.DEVICE}

@app.post("/predict-image", response_model=PredictImageResponse)
@torch.no_grad()
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    tensor = preprocess_image(contents)

    if settings.DEVICE == 'cuda':
        tensor = tensor.cuda()

    outputs = model(tensor)[0]
    boxes, labels, scores = filter_predictions(outputs, settings.SCORE_THRESH)

    detections = []
    for b, l, s in zip(boxes, labels, scores):
        detections.append(Box(
            x_min=float(b[0]), y_min=float(b[1]), x_max=float(b[2]), y_max=float(b[3]),
            score=float(s), label_id=int(l)
        ))

    return PredictImageResponse(detections=detections, num_detections=len(detections))