import io
import cv2
import torch
import threading
import queue
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from PIL import Image, ImageDraw, ImageFont
from app.model import load_model
from app.predict_utils import preprocess_image, filter_predictions
from app.config import settings

# ============================================================
#  Initialization
# ============================================================
app = FastAPI(title="Faster R-CNN API (Optimized Live Stream)")
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

model = None
device = torch.device(settings.DEVICE)
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

# ============================================================
#  Model Loader
# ============================================================
@app.on_event("startup")
def startup_event():
    global model
    torch.set_num_threads(settings.NUM_THREADS)
    model = load_model()
    model.eval()
    print(f"✅ Model loaded successfully on {device}")

    # Start background inference thread
    threading.Thread(target=inference_worker, daemon=True).start()


# ============================================================
#  Inference Worker (runs in background thread)
# ============================================================
def inference_worker():
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((320, 320)),  # smaller = faster
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(img_rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor)[0]

            keep = outputs["scores"] >= settings.SCORE_THRESH
            boxes = outputs["boxes"][keep].cpu().numpy()
            labels = outputs["labels"][keep].cpu().numpy()
            scores = outputs["scores"][keep].cpu().numpy()

            result_queue.put((boxes, labels, scores))
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            continue


# ============================================================
#  Health Check
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


# ============================================================
#  Image Prediction Endpoint
# ============================================================
@app.post("/predict-image")
@torch.no_grad()
def predict_image(file: UploadFile = File(...)):
    contents = file.file.read()
    tensor = preprocess_image(contents)
    if device.type != "cpu":
        tensor = tensor.to(device)

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


# ============================================================
#  Visualization Endpoint
# ============================================================
@app.post("/visualize-image")
@torch.no_grad()
def visualize_image(file: UploadFile = File(...)):
    contents = file.file.read()
    tensor = preprocess_image(contents)
    if device.type != "cpu":
        tensor = tensor.to(device)

    outputs = model(tensor)[0]
    boxes, labels, scores = filter_predictions(outputs, settings.SCORE_THRESH)

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for (b, l, s) in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = map(float, b)
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, max(y_min - 10, 0)), f"ID:{int(l)} | {s:.2f}", fill="yellow", font=font)

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/jpeg")


# ============================================================
#  Live Stream Page
# ============================================================
@app.get("/live", response_class=HTMLResponse)
def live_stream():
    return """
    <html>
        <head>
            <title>📸 Faster R-CNN Live Stream</title>
            <style>
                body { background: #111; color: white; text-align: center; }
                img { border-radius: 10px; width: 80%; margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>Live Faster R-CNN Detection</h1>
            <p>Press Ctrl+C in terminal to stop</p>
            <img src="/video-feed" />
        </body>
    </html>
    """


# ============================================================
#  Video Feed Generator (Optimized)
# ============================================================
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access camera")

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Skip frames to maintain smooth FPS
        frame_count += 1
        if frame_count % 3 != 0:
            continue

        if not frame_queue.full():
            frame_queue.put(frame)

        if not result_queue.empty():
            boxes, labels, scores = result_queue.get_nowait()
            for (b, l, s) in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{int(l)} {s:.2f}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/video-feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")