import io
import os
import cv2
import torch
import threading
import queue
import tempfile
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from PIL import Image, ImageDraw, ImageFont
from app.model import load_model
from app.predict_utils import preprocess_image, filter_predictions
from app.config import settings
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request
import json
from datetime import datetime

# ============================================================
# Initialization
# ============================================================
app = FastAPI(title="Faster R-CNN API (Optimized Live Stream)")
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

templates = Jinja2Templates(directory="app/templates")
RESULTS_DIR = os.getenv("RESULTS_DIR", "/results")

model = None
device = torch.device(settings.DEVICE)
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://gd-live.com", "http://localhost:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Model Loader
# ============================================================
@app.on_event("startup")
def startup_event():
    global model
    torch.set_num_threads(settings.NUM_THREADS)
    model = load_model()
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
    threading.Thread(target=inference_worker, daemon=True).start()

# ============================================================
# Inference Worker
# ============================================================
def inference_worker():
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((320, 320)),
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

# ============================================================
# Health Check
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

# ============================================================
# Save Image and Json File After Analysis
# ============================================================

def save_analysis(image_bytes, detections):
    """Save image and detections under RESULTS_DIR."""
    img_dir = os.path.join(RESULTS_DIR, "images")
    json_dir = os.path.join(RESULTS_DIR, "json")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(img_dir, f"{ts}.jpg")
    json_path = os.path.join(json_dir, f"{ts}.json")
    
    with open(img_path, "wb") as f:
        f.write(image_bytes)
    with open(json_path, "w") as f:
        import json
        json.dump(detections, f, indent=2)
    
    print(f"✅ Saved: {img_path} and {json_path}")

def save_coco_format(image_bytes, detections, image_filename="captured.jpg"):
    """Save detections in COCO format for retraining."""
    coco_dir = os.path.join(RESULTS_DIR, "coco")
    os.makedirs(coco_dir, exist_ok=True)

    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"results/coco/{ts}.json"

    # Get image dimensions
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size

    coco_data = {
        "info": {
            "description": "Auto-collected detections from Faster R-CNN API",
            "version": "1.0",
            "date_created": ts
        },
        "images": [
            {
                "id": 1,
                "file_name": image_filename,
                "width": width,
                "height": height
            }
        ],
        "annotations": [],
        "categories": []
    }

    annotation_id = 1
    label_set = set()

    for det in detections["detections"]:
        x_min = det["x_min"]
        y_min = det["y_min"]
        width_box = det["x_max"] - det["x_min"]
        height_box = det["y_max"] - det["y_min"]
        score = det["score"]
        label_id = det["label_id"]

        coco_data["annotations"].append({
            "id": annotation_id,
            "image_id": 1,
            "category_id": label_id,
            "bbox": [x_min, y_min, width_box, height_box],
            "area": width_box * height_box,
            "iscrowd": 0,
            "score": score
        })
        label_set.add(label_id)
        annotation_id += 1

    coco_data["categories"] = [{"id": int(l), "name": f"class_{l}"} for l in sorted(label_set)]

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"✅ COCO file saved: {json_path}")

# ============================================================
# Predict Single Image
# ============================================================
@app.post("/predict-image")
@torch.no_grad()
def predict_image(file: UploadFile = File(...)):
    try:
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

        # ✅ Save both image and detection results
        save_analysis(contents, {"detections": detections, "num_detections": len(detections)})
        save_coco_format(contents, {"detections": detections, "num_detections": len(detections)}, file.filename)

        return {"detections": detections, "num_detections": len(detections)}
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {"detections": [], "num_detections": 0, "error": str(e)}

# ============================================================
# Visualize Image Endpoint
# ============================================================
@app.post("/visualize-image")
@torch.no_grad()
def visualize_image(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        tensor = preprocess_image(contents)
        if device.type != "cpu":
            tensor = tensor.to(device)
        outputs = model(tensor)[0]
        boxes, labels, scores = filter_predictions(outputs, settings.SCORE_THRESH)
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for b, l, s in zip(boxes, labels, scores):
            x_min, y_min, x_max, y_max = map(float, b)
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            draw.text((x_min, max(y_min - 10, 0)),
                      f"ID:{int(l)} | {s:.2f}", fill="yellow", font=font)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)
        return StreamingResponse(img_bytes, media_type="image/jpeg")
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return {"error": str(e)}
    

# ============================================================
# Predict Video/Image Endpoint
# ============================================================
@app.get("/", response_class=HTMLResponse)
def predict_classes(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ============================================================
# Live Stream Web Page
# ============================================================
@app.get("/live", response_class=HTMLResponse)
def live_page(request: Request):
    return templates.TemplateResponse("live.html", {"request": request})

# ============================================================
# Optimized Video Feed
# ============================================================
def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access camera")

    frame_count = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            
            if frame_count % 3 != 0:
                continue
            
            if not frame_queue.full():
                frame_queue.put(frame.copy())
            
            if not result_queue.empty():
                try:
                    boxes, labels, scores = result_queue.get_nowait()
                    for b, l, s in zip(boxes, labels, scores):
                        x1, y1, x2, y2 = map(int, b)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"{int(l)} {s:.2f}", (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except queue.Empty:
                    pass
            
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

@app.get("/video-feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

# ============================================================
# Analyze Uploaded Video
# ============================================================
@app.post("/analyze-video")
def analyze_video(file: UploadFile = File(...)):
    allowed_ext = (".mp4", ".avi", ".mov", ".mkv")
    if not any(file.filename.lower().endswith(ext) for ext in allowed_ext):
        return {"error": "Unsupported file type. Allowed: .mp4, .avi, .mov, .mkv"}

    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix="." + file.filename.split(".")[-1])
    tmp_in.write(file.file.read())
    tmp_in.close()
    tmp_out = tempfile.mktemp(suffix=".mp4")

    cap = None
    out = None
    try:
        cap = cv2.VideoCapture(tmp_in.name)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_out, fourcc, fps, (w, h))

        transform = T.Compose([
            T.ToTensor(),
            T.Resize((320, 320)),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(img_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)[0]

            keep = outputs["scores"] >= settings.SCORE_THRESH
            boxes = outputs["boxes"][keep].cpu().numpy()
            labels = outputs["labels"][keep].cpu().numpy()
            scores = outputs["scores"][keep].cpu().numpy()

            for b, l, s in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{int(l)}:{s:.2f}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            out.write(frame)
            
            if frame_idx % 30 == 0:
                print(f"Processing frame {frame_idx}...")

        print(f"✅ Processed {frame_idx} frames successfully")
        
        cap.release()
        out.release()
        
        return StreamingResponse(open(tmp_out, "rb"), media_type="video/mp4")
    
    except Exception as e:
        print(f"❌ Video analysis error: {e}")
        return {"error": str(e)}
    finally:
        if cap:
            cap.release()
        if out:
            out.release()
        try:
            os.unlink(tmp_in.name)
        except:
            pass