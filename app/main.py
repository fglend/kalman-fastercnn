import io
import cv2
import torch
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from PIL import Image, ImageDraw, ImageFont
from app.model import load_model
from app.predict_utils import preprocess_image, filter_predictions
from app.config import settings

# -----------------------------------------
# Initialize API and Model
# -----------------------------------------
app = FastAPI(title="Faster R-CNN API with Visualization and Live Stream")
model = None


@app.on_event("startup")
def _startup():
    """Load model at startup"""
    global model
    torch.set_num_threads(settings.NUM_THREADS)
    model = load_model()
    print(f"✅ Model loaded on {settings.DEVICE}")


@app.get("/health")
def health():
    """Check server and device health"""
    return {"status": "ok", "device": settings.DEVICE}


# -----------------------------------------
# IMAGE INFERENCE ENDPOINT
# -----------------------------------------
@app.post("/predict-image")
@torch.no_grad()
def predict_image(file: UploadFile = File(...)):
    """Perform prediction and return bounding box data"""
    contents = file.file.read()
    tensor = preprocess_image(contents)

    if settings.DEVICE == "cuda":
        tensor = tensor.cuda()

    outputs = model(tensor)[0]
    boxes, labels, scores = filter_predictions(outputs, settings.SCORE_THRESH)

    detections = [
        {
            "x_min": float(b[0]),
            "y_min": float(b[1]),
            "x_max": float(b[2]),
            "y_max": float(b[3]),
            "score": float(s),
            "label_id": int(l),
        }
        for b, l, s in zip(boxes, labels, scores)
    ]

    return {"detections": detections, "num_detections": len(detections)}


# -----------------------------------------
# VISUALIZATION ENDPOINT
# -----------------------------------------
@app.post("/visualize-image")
@torch.no_grad()
def visualize_image(file: UploadFile = File(...)):
    """Return an image with drawn bounding boxes"""
    contents = file.file.read()
    tensor = preprocess_image(contents)

    if settings.DEVICE == "cuda":
        tensor = tensor.cuda()

    outputs = model(tensor)[0]
    boxes, labels, scores = filter_predictions(outputs, settings.SCORE_THRESH)

    # Load image and draw boxes
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    for (b, l, s) in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = map(float, b)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, max(y_min - 10, 0)), f"ID:{int(l)} | {s:.2f}", fill="yellow", font=font)

    # Convert image to streamable bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/jpeg")


# -----------------------------------------
# LIVE CAMERA STREAM ENDPOINT
# -----------------------------------------
@app.get("/live", response_class=HTMLResponse)
def live_stream():
    """Simple web page for live detection"""
    return """
    <html>
        <head>
            <title>📸 Live Faster R-CNN Stream</title>
            <style>
                body { background: #111; color: white; text-align: center; }
                img { border-radius: 10px; width: 80%; margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>Faster R-CNN Real-time Detection</h1>
            <img src="/video-feed" />
        </body>
    </html>
    """


def generate_frames():
    """Read webcam and yield frames with bounding boxes"""
    cap = cv2.VideoCapture(0)  # For external cam, use index 1 or RTSP/USB path
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access camera")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    while True:
        success, frame = cap.read()
        if not success:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0).to(settings.DEVICE)

        with torch.no_grad():
            outputs = model(tensor)[0]

        keep = outputs["scores"] >= settings.SCORE_THRESH
        boxes = outputs["boxes"][keep].cpu().numpy()
        labels = outputs["labels"][keep].cpu().numpy()
        scores = outputs["scores"][keep].cpu().numpy()

        for (box, label, score) in zip(boxes, labels, scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{int(label)} {score:.2f}", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/video-feed")
def video_feed():
    """Stream the live webcam feed"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")