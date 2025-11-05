import io
import os
import cv2
import torch
import threading
import queue
import tempfile
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File, Response
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
import threading
import uuid
import time
from threading import Thread
from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse
from PIL import ImageEnhance, ImageOps

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
# Model Loader - Load Both Models from .pth files
# ============================================================
def load_model_from_checkpoint(checkpoint_path, backbone="resnet50", num_classes=7):
    """Load a trained Faster R-CNN FPN model from checkpoint."""
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    
    # Create model based on backbone using torchvision's FPN models
    print(f"Creating Faster R-CNN with {backbone.upper()} + FPN backbone...")
    
    if backbone == "resnet50":
        # Use the standard FasterRCNN ResNet50 FPN model
        try:
            # Try to use torchvision's built-in model
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=num_classes)
        except Exception as e:
            print(f"⚠️ Warning: Could not create model with standard method: {e}")
            # Fallback to custom creation
            from torchvision.models.detection.faster_rcnn import FasterRCNN
            from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
            backbone_model = resnet_fpn_backbone('resnet50', weights=None)
            model = FasterRCNN(backbone_model, num_classes=num_classes)
            
    elif backbone == "resnet101":
        # Use FasterRCNN ResNet101 FPN model
        try:
            from torchvision.models.detection.faster_rcnn import FasterRCNN
            from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
            backbone_model = resnet_fpn_backbone('resnet101', weights=None)
            model = FasterRCNN(backbone_model, num_classes=num_classes)
        except Exception as e:
            print(f"⚠️ Warning: Error creating ResNet101 model: {e}")
            raise
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"📋 Found 'model_state_dict' (Epoch: {checkpoint.get('epoch', 'N/A')})")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"📋 Found 'state_dict'")
        else:
            state_dict = checkpoint
            print(f"📋 Using checkpoint as state_dict")
    else:
        state_dict = checkpoint
        print(f"📋 Checkpoint is direct state_dict")
    
    # Load state dict with strict=False to allow minor mismatches
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ Missing keys: {len(missing_keys)} (this is usually okay)")
            if len(missing_keys) < 10:
                for key in missing_keys[:5]:
                    print(f"   - {key}")
        
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {len(unexpected_keys)} (this is usually okay)")
            if len(unexpected_keys) < 10:
                for key in unexpected_keys[:5]:
                    print(f"   - {key}")
        
        print(f"✅ Checkpoint loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading state dict: {e}")
        raise
    
    model.to(device)
    model.eval()
    return model

@app.on_event("startup")
def startup_event():
    global model_resnet50, model_resnet101
    torch.set_num_threads(settings.NUM_THREADS)
    
    # Define paths to your .pth files
    RESNET50_PATH = os.getenv("RESNET50_MODEL_PATH", "models/faster_rcnn_resnet50.pth")
    RESNET101_PATH = os.getenv("RESNET101_MODEL_PATH", "models/faster_rcnn_resnet101.pth")
    NUM_CLASSES = int(os.getenv("NUM_CLASSES", "7"))  # Adjust based on your dataset
    
    # Load ResNet50 model
    print("🔄 Loading Faster-RCNN model...")
    try:
        model_resnet50 = load_model_from_checkpoint(
            RESNET50_PATH, 
            backbone="resnet50",
            num_classes=NUM_CLASSES
        )
         # ✅ Allow more detections per image
        model_resnet50.roi_heads.detections_per_img = 350
        model_resnet50.rpn.post_nms_top_n_test = 2000
        model_resnet50.rpn.pre_nms_top_n_test = 2000
        print(f"✅ Faster-RCNN model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load ResNet50: {e}")
        raise
    
    # Load ResNet101 model
    print("🔄 Loading Hybrid model...")
    try:
        model_resnet101 = load_model_from_checkpoint(
            RESNET101_PATH,
            backbone="resnet101", 
            num_classes=NUM_CLASSES
        )
        model_resnet101.roi_heads.detections_per_img = 350
        model_resnet101.rpn.post_nms_top_n_test = 2000
        model_resnet101.rpn.pre_nms_top_n_test = 2000
        print(f"✅ Hybrid model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load ResNet101: {e}")
        raise
    
    print(f"✅ Both models loaded successfully on {device}")


# ============================================================
# Inference Worker
# ============================================================
def inference_worker():
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((512, 512)),
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
    return {
        "status": "ok", 
        "device": str(device),
        "models": ["resnet50", "resnet101"]
    }


# =========================
# JOB STORAGE
# =========================
jobs = {}  # job_id -> {"status": str, "progress": float, "output": str, "error": str}

def process_video_job(job_id, video_path, output_path):
    import cv2, torch, torchvision.transforms as T
    try:
        jobs[job_id]["status"] = "processing"
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w, h = int(cap.get(3)), int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        transform = T.Compose([
            T.ToTensor(),
            T.Resize((384, 384)),  # ✅ Optimized size
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        frame_i = 0
        skip = 5  # ✅ Increased for speed
        detections_summary = []  # ✅ Track detections

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_i += 1
            if frame_i % skip != 0:
                out.write(frame)  # Write unprocessed frames
                continue

            # ✅ Apply grayscale + darken
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            darkened = cv2.convertScaleAbs(gray, alpha=0.8, beta=0)
            frame = cv2.cvtColor(darkened, cv2.COLOR_GRAY2BGR)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(img_rgb).unsqueeze(0).to(device, non_blocking=True)
            
            # ✅ FP16 for GPU
            if device.type == "cuda":
                tensor = tensor.half()
            
            with torch.no_grad():
                outputs = model(tensor)[0]

            keep = outputs["scores"] >= settings.SCORE_THRESH
            boxes = outputs["boxes"][keep].cpu().numpy()
            labels = outputs["labels"][keep].cpu().numpy()
            scores = outputs["scores"][keep].cpu().numpy()
            
            # ✅ Track detection count
            if len(boxes) > 0:
                detections_summary.append({
                    "frame": frame_i,
                    "count": len(boxes)
                })
            
            sx, sy = w/384, h/384  # ✅ Updated scale
            for b, l, s in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, [b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{int(l)}:{s:.2f}", (x1, max(20, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            out.write(frame)

            if frame_i % 10 == 0:
                jobs[job_id]["progress"] = round((frame_i / total) * 100, 2)

        cap.release()
        out.release()
        
        # ✅ Save to permanent storage
        video_dir = os.path.join(RESULTS_DIR, "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        permanent_path = os.path.join(video_dir, f"{ts}.mp4")
        
        # Move temp file to permanent location
        import shutil
        shutil.move(output_path, permanent_path)
        
        # ✅ Save metadata
        video_json_dir = os.path.join(RESULTS_DIR, "videos_json")
        os.makedirs(video_json_dir, exist_ok=True)
        
        metadata = {
            "timestamp": ts,
            "total_frames": total,
            "processed_frames": frame_i,
            "fps": fps,
            "width": w,
            "height": h,
            "total_detections": sum(d["count"] for d in detections_summary),
            "frames_with_detections": len(detections_summary),
            "detection_summary": detections_summary[:100]  # First 100 frames
        }
        
        with open(os.path.join(video_json_dir, f"{ts}.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Video saved: {permanent_path}")
        jobs[job_id].update({
            "status": "done", 
            "progress": 100.0, 
            "output": permanent_path,
            "timestamp": ts  # ✅ Add timestamp
        })
        
    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e)})

@app.post("/start-analyze")
def start_analyze(file: UploadFile = File(...)):
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix="." + file.filename.split(".")[-1])
    tmp_in.write(file.file.read())
    tmp_in.close()
    tmp_out = tempfile.mktemp(suffix=".mp4")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": 0.0, "output": tmp_out, "error": ""}

    thread = Thread(target=process_video_job, args=(job_id, tmp_in.name, tmp_out), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/progress/{job_id}")
def get_progress(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse(content={"error": "Job not found"}, status_code=404)
    return {"status": job["status"], "progress": job["progress"], "error": job["error"]}


@app.get("/result/{job_id}")
def get_result(job_id: str):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return JSONResponse(content={"error": "Job not ready"}, status_code=400)
    
    # ✅ Add timestamp to response headers
    response = StreamingResponse(open(job["output"], "rb"), media_type="video/mp4")
    if "timestamp" in job:
        response.headers["X-Video-Timestamp"] = job["timestamp"]
    return response

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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(coco_dir, f"{ts}.json")

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

        outputs = model_resnet101(tensor)[0]
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

        # ✅ Always compute and save with num_detections
        result_data = {
            "detections": detections,
            "num_detections": len(detections)
        }

        # save_analysis(contents, result_data)
        # save_coco_format(contents, result_data, file.filename)

             # ✅ Run saving in background thread
        def async_save():
            try:
                save_analysis(contents, result_data)
                save_coco_format(contents, result_data, file.filename)
            except Exception as e:
                print(f"⚠️ Async save error: {e}")

        threading.Thread(target=async_save, daemon=True).start()

        print(f"✅ {len(detections)} objects detected")
        return result_data

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

        outputs = model_resnet101(tensor)[0]
        boxes, labels, scores = filter_predictions(outputs, settings.SCORE_THRESH)

        # ✅ Convert to grayscale and darken
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = ImageOps.grayscale(image)  # Convert to black and white
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.8)
        image = image.convert("RGB")

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

        # ✅ Optimize for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tmp_out, fourcc, fps, (w, h))

        # ✅ Define transform once
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((512, 512)),  # match training
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        frame_idx = 0
        skip_frames = 5  # ✅ skip every other frame for speed (~2× faster)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"🎥 Processing {total_frames} frames (skipping {skip_frames-1} of each {skip_frames})")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % skip_frames != 0:
                continue  # ✅ skip frames to reduce load

             # ✅ Apply grayscale + darken
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            darkened = cv2.convertScaleAbs(gray, alpha=0.8, beta=0)
            frame = cv2.cvtColor(darkened, cv2.COLOR_GRAY2BGR)

            # ✅ Resize to smaller copy for processing, keep original for draw
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = transform(img_rgb).unsqueeze(0).to(device, non_blocking=True)

            with torch.no_grad():
                outputs = model(tensor)[0]

            keep = outputs["scores"] >= settings.SCORE_THRESH
            boxes = outputs["boxes"][keep].cpu().numpy()
            labels = outputs["labels"][keep].cpu().numpy()
            scores = outputs["scores"][keep].cpu().numpy()

            # ✅ Scale boxes back to original size
            scale_x = w / 512
            scale_y = h / 512
            for b, l, s in zip(boxes, labels, scores):
                x1, y1, x2, y2 = (int(b[0]*scale_x), int(b[1]*scale_y),
                                  int(b[2]*scale_x), int(b[3]*scale_y))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{int(l)}:{s:.2f}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            out.write(frame)

            # ✅ Progress print every 60 processed frames
            if frame_idx % (60 * skip_frames) == 0:
                print(f"Processed {frame_idx}/{total_frames} frames...")

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

# ============================================================
# Gallery Endpoints
# ============================================================

@app.get("/gallery/list")
def list_gallery():
    """List all saved detections with metadata."""
    try:
        json_dir = os.path.join(RESULTS_DIR, "json")
        if not os.path.exists(json_dir):
            return {"items": []}
        
        items = []
        for filename in os.listdir(json_dir):
            if filename.endswith(".json"):
                timestamp = filename.replace(".json", "")
                json_path = os.path.join(json_dir, filename)
                
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    
                    items.append({
                        "timestamp": timestamp,
                        "num_detections": data.get("num_detections", 0)
                    })
                except Exception as e:
                    print(f"⚠️ Error reading {filename}: {e}")
        
        # Sort by timestamp (newest first)
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"items": items}
    
    except Exception as e:
        print(f"❌ Gallery list error: {e}")
        return {"items": [], "error": str(e)}


@app.get("/gallery/image/{timestamp}")
def get_gallery_image(timestamp: str):
    """Get original image by timestamp."""
    try:
        img_path = os.path.join(RESULTS_DIR, "images", f"{timestamp}.jpg")
        if not os.path.exists(img_path):
            return JSONResponse(content={"error": "Image not found"}, status_code=404)
        
        return StreamingResponse(open(img_path, "rb"), media_type="image/jpeg")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/gallery/visualize/{timestamp}")
def visualize_gallery_image(timestamp: str):
    """Generate visualized image with bounding boxes on-the-fly."""
    try:
        img_path = os.path.join(RESULTS_DIR, "images", f"{timestamp}.jpg")
        json_path = os.path.join(RESULTS_DIR, "json", f"{timestamp}.json")
        
        if not os.path.exists(img_path) or not os.path.exists(json_path):
            return JSONResponse(content={"error": "Files not found"}, status_code=404)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        
        # Load detections
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Draw bounding boxes
        for det in data.get("detections", []):
            x_min = det["x_min"]
            y_min = det["y_min"]
            x_max = det["x_max"]
            y_max = det["y_max"]
            score = det["score"]
            label_id = det["label_id"]
            
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            draw.text((x_min, max(y_min - 10, 0)),
                      f"ID:{label_id} | {score:.2f}", fill="yellow", font=font)
        
        # Return as JPEG
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG", quality=95)
        img_bytes.seek(0)
        
        return StreamingResponse(img_bytes, media_type="image/jpeg")
    
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/gallery/json/{timestamp}")
def get_gallery_json(timestamp: str):
    """Get detection JSON data by timestamp."""
    try:
        json_path = os.path.join(RESULTS_DIR, "json", f"{timestamp}.json")
        if not os.path.exists(json_path):
            return JSONResponse(content={"error": "JSON not found"}, status_code=404)
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.delete("/gallery/delete/{timestamp}")
def delete_gallery_item(timestamp: str):
    """Delete a gallery item (image, json, and coco files)."""
    try:
        deleted = []
        
        # Delete image
        img_path = os.path.join(RESULTS_DIR, "images", f"{timestamp}.jpg")
        if os.path.exists(img_path):
            os.remove(img_path)
            deleted.append("image")
        
        # Delete JSON
        json_path = os.path.join(RESULTS_DIR, "json", f"{timestamp}.json")
        if os.path.exists(json_path):
            os.remove(json_path)
            deleted.append("json")
        
        # Delete COCO
        coco_path = os.path.join(RESULTS_DIR, "coco", f"{timestamp}.json")
        if os.path.exists(coco_path):
            os.remove(coco_path)
            deleted.append("coco")
        
        if deleted:
            print(f"✅ Deleted {timestamp}: {', '.join(deleted)}")
            return {"success": True, "deleted": deleted}
        else:
            return JSONResponse(content={"error": "Item not found"}, status_code=404)
    
    except Exception as e:
        print(f"❌ Delete error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# ============================================================
# Video Gallery Endpoints
# ============================================================
@app.get("/gallery/videos/list")
def list_video_gallery():
    """List all saved analyzed videos."""
    try:
        video_json_dir = os.path.join(RESULTS_DIR, "videos_json")
        if not os.path.exists(video_json_dir):
            return {"items": []}
        
        items = []
        for filename in os.listdir(video_json_dir):
            if filename.endswith(".json"):
                timestamp = filename.replace(".json", "")
                json_path = os.path.join(video_json_dir, filename)
                
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    
                    items.append({
                        "timestamp": timestamp,
                        "total_detections": data.get("total_detections", 0),
                        "frames_with_detections": data.get("frames_with_detections", 0),
                        "total_frames": data.get("total_frames", 0),
                        "fps": data.get("fps", 0),
                        "duration": round(data.get("total_frames", 0) / max(data.get("fps", 1), 1), 1)
                    })
                except Exception as e:
                    print(f"⚠️ Error reading {filename}: {e}")
        
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"items": items}
    
    except Exception as e:
        print(f"❌ Video gallery list error: {e}")
        return {"items": [], "error": str(e)}


@app.get("/gallery/videos/stream/{timestamp}")
def stream_gallery_video(timestamp: str):
    """Stream analyzed video by timestamp."""
    try:
        video_path = os.path.join(RESULTS_DIR, "videos", f"{timestamp}.mp4")
        if not os.path.exists(video_path):
            return JSONResponse(content={"error": "Video not found"}, status_code=404)
        
        return StreamingResponse(open(video_path, "rb"), media_type="video/mp4")
    except Exception as e:
        print(f"❌ Error streaming video: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/gallery/videos/thumbnail/{timestamp}")
def get_video_thumbnail(timestamp: str):
    """Generate thumbnail for video (first frame with detections)."""
    try:
        video_path = os.path.join(RESULTS_DIR, "videos", f"{timestamp}.mp4")
        if not os.path.exists(video_path):
            return JSONResponse(content={"error": "Video not found"}, status_code=404)
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return JSONResponse(content={"error": "Cannot read video"}, status_code=500)
        
        # Resize for thumbnail
        h, w = frame.shape[:2]
        thumb_w = 400
        thumb_h = int(h * (thumb_w / w))
        frame = cv2.resize(frame, (thumb_w, thumb_h))
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
    
    except Exception as e:
        print(f"❌ Thumbnail error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/gallery/videos/json/{timestamp}")
def get_video_metadata(timestamp: str):
    """Get video metadata."""
    try:
        json_path = os.path.join(RESULTS_DIR, "videos_json", f"{timestamp}.json")
        if not os.path.exists(json_path):
            return JSONResponse(content={"error": "Metadata not found"}, status_code=404)
        
        with open(json_path, "r") as f:
            data = json.load(f)
        
        return data
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.delete("/gallery/videos/delete/{timestamp}")
def delete_video_item(timestamp: str):
    """Delete a video gallery item."""
    try:
        deleted = []
        
        video_path = os.path.join(RESULTS_DIR, "videos", f"{timestamp}.mp4")
        if os.path.exists(video_path):
            os.remove(video_path)
            deleted.append("video")
        
        json_path = os.path.join(RESULTS_DIR, "videos_json", f"{timestamp}.json")
        if os.path.exists(json_path):
            os.remove(json_path)
            deleted.append("json")
        
        if deleted:
            print(f"✅ Deleted video {timestamp}")
            return {"success": True, "deleted": deleted}
        else:
            return JSONResponse(content={"error": "Video not found"}, status_code=404)
    
    except Exception as e:
        print(f"❌ Delete error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# ============================================================
# Compare Models on Single Image
# ============================================================
@app.post("/compare-models")
@torch.no_grad()
def compare_models(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        tensor = preprocess_image(contents)
        if device.type != "cpu":
            tensor = tensor.to(device)

        # Run both models
        outputs_resnet50 = model_resnet50(tensor)[0]
        outputs_resnet101 = model_resnet101(tensor)[0]

        # Filter predictions for both
        boxes_50, labels_50, scores_50 = filter_predictions(outputs_resnet50, settings.SCORE_THRESH)
        boxes_101, labels_101, scores_101 = filter_predictions(outputs_resnet101, settings.SCORE_THRESH)

        # Create detections for both
        detections_resnet50 = [
            {
                "x_min": float(b[0]),
                "y_min": float(b[1]),
                "x_max": float(b[2]),
                "y_max": float(b[3]),
                "score": float(s),
                "label_id": int(l),
            }
            for b, l, s in zip(boxes_50, labels_50, scores_50)
        ]

        detections_resnet101 = [
            {
                "x_min": float(b[0]),
                "y_min": float(b[1]),
                "x_max": float(b[2]),
                "y_max": float(b[3]),
                "score": float(s),
                "label_id": int(l),
            }
            for b, l, s in zip(boxes_101, labels_101, scores_101)
        ]

        result_data = {
            "resnet50": {
                "detections": detections_resnet50,
                "num_detections": len(detections_resnet50)
            },
            "resnet101": {
                "detections": detections_resnet101,
                "num_detections": len(detections_resnet101)
            }
        }

        # Save comparison results
        def async_save():
            try:
                save_comparison(contents, result_data, file.filename)
            except Exception as e:
                print(f"⚠️ Async save error: {e}")

        threading.Thread(target=async_save, daemon=True).start()

        print(f"✅ Stand-Alone: {len(detections_resnet50)} objects | Hybrid: {len(detections_resnet101)} objects")
        return result_data

    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return {
            "resnet50": {"detections": [], "num_detections": 0},
            "resnet101": {"detections": [], "num_detections": 0},
            "error": str(e)
        }

# ============================================================
# Visualize Both Models Side by Side
# ============================================================
@app.post("/visualize-comparison")
@torch.no_grad()
def visualize_comparison(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        tensor = preprocess_image(contents)
        if device.type != "cpu":
            tensor = tensor.to(device)

        # Run both models
        outputs_resnet50 = model_resnet50(tensor)[0]
        outputs_resnet101 = model_resnet101(tensor)[0]

        boxes_50, labels_50, scores_50 = filter_predictions(outputs_resnet50, settings.SCORE_THRESH)
        boxes_101, labels_101, scores_101 = filter_predictions(outputs_resnet101, settings.SCORE_THRESH)

        # Load and prepare images
        image_50 = Image.open(io.BytesIO(contents)).convert("RGB")
        image_101 = image_50.copy()

        # Apply grayscale + darken
        image_50 = ImageOps.grayscale(image_50)
        enhancer = ImageEnhance.Brightness(image_50)
        image_50 = enhancer.enhance(0.8).convert("RGB")

        image_101 = ImageOps.grayscale(image_101)
        enhancer = ImageEnhance.Brightness(image_101)
        image_101 = enhancer.enhance(0.8).convert("RGB")

        # Draw on ResNet50 image
        draw_50 = ImageDraw.Draw(image_50)
        font = ImageFont.load_default()
        for b, l, s in zip(boxes_50, labels_50, scores_50):
            x_min, y_min, x_max, y_max = map(float, b)
            draw_50.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            draw_50.text((x_min, max(y_min - 10, 0)),
                        f"ID:{int(l)} | {s:.2f}", fill="yellow", font=font)

        # Draw on ResNet101 image
        draw_101 = ImageDraw.Draw(image_101)
        for b, l, s in zip(boxes_101, labels_101, scores_101):
            x_min, y_min, x_max, y_max = map(float, b)
            draw_101.rectangle([x_min, y_min, x_max, y_max], outline="lime", width=3)
            draw_101.text((x_min, max(y_min - 10, 0)),
                         f"ID:{int(l)} | {s:.2f}", fill="cyan", font=font)

        # Create side-by-side image
        width, height = image_50.size
        combined = Image.new('RGB', (width * 2 + 20, height + 60), color=(20, 20, 30))
        
        # Add labels
        draw_combined = ImageDraw.Draw(combined)
        title_font = ImageFont.load_default()
        draw_combined.text((width // 2 - 50, 10), "Stand-Alone", fill="red", font=title_font)
        draw_combined.text((width + width // 2 - 30, 10), "Hybrid", fill="lime", font=title_font)
        
        # Paste images
        combined.paste(image_50, (0, 40))
        combined.paste(image_101, (width + 20, 40))

        # Return combined image
        img_bytes = io.BytesIO()
        combined.save(img_bytes, format="JPEG", quality=95)
        img_bytes.seek(0)
        return StreamingResponse(img_bytes, media_type="image/jpeg")

    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ============================================================
# Save Comparison Results
# ============================================================
def save_comparison(image_bytes, comparison_data, image_filename="comparison.jpg"):
    """Save comparison results for both models."""
    comp_dir = os.path.join(RESULTS_DIR, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(comp_dir, f"{ts}.json")
    img_path = os.path.join(comp_dir, f"{ts}.jpg")

    # Save image
    with open(img_path, "wb") as f:
        f.write(image_bytes)

    # Save comparison JSON
    comparison_data["timestamp"] = ts
    comparison_data["filename"] = image_filename
    
    with open(json_path, "w") as f:
        json.dump(comparison_data, f, indent=2)

    print(f"✅ Comparison saved: {json_path}")

# ============================================================
# Get Comparison History
# ============================================================
@app.get("/comparisons/list")
def list_comparisons():
    """List all saved model comparisons."""
    try:
        comp_dir = os.path.join(RESULTS_DIR, "comparisons")
        if not os.path.exists(comp_dir):
            return {"items": []}
        
        items = []
        for filename in os.listdir(comp_dir):
            if filename.endswith(".json"):
                timestamp = filename.replace(".json", "")
                json_path = os.path.join(comp_dir, filename)
                
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    
                    items.append({
                        "timestamp": timestamp,
                        "resnet50_detections": data.get("resnet50", {}).get("num_detections", 0),
                        "resnet101_detections": data.get("resnet101", {}).get("num_detections", 0)
                    })
                except Exception as e:
                    print(f"⚠️ Error reading {filename}: {e}")
        
        items.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"items": items}
    
    except Exception as e:
        print(f"❌ Comparison list error: {e}")
        return {"items": [], "error": str(e)}


@app.get("/comparisons/image/{timestamp}")
def get_comparison_image(timestamp: str):
    """Stream the saved original image for a comparison item."""
    img_path = os.path.join(RESULTS_DIR, "comparisons", f"{timestamp}.jpg")
    if not os.path.exists(img_path):
        return JSONResponse(content={"error": "Image not found"}, status_code=404)
    return StreamingResponse(open(img_path, "rb"), media_type="image/jpeg")

@app.get("/comparisons/json/{timestamp}")
def get_comparison_json(timestamp: str):
    """Return the saved comparison JSON."""
    json_path = os.path.join(RESULTS_DIR, "comparisons", f"{timestamp}.json")
    if not os.path.exists(json_path):
        return JSONResponse(content={"error": "JSON not found"}, status_code=404)
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

@app.delete("/comparisons/delete/{timestamp}")
def delete_comparison_item(timestamp: str):
    """Delete a comparison (image + json)."""
    deleted = []
    img_path = os.path.join(RESULTS_DIR, "comparisons", f"{timestamp}.jpg")
    json_path = os.path.join(RESULTS_DIR, "comparisons", f"{timestamp}.json")

    if os.path.exists(img_path):
        os.remove(img_path)
        deleted.append("image")
    if os.path.exists(json_path):
        os.remove(json_path)
        deleted.append("json")

    if deleted:
        return {"success": True, "deleted": deleted}
    return JSONResponse(content={"error": "Item not found"}, status_code=404)

# ============================================================
# Comparison Page Route
# ============================================================
@app.get("/comparison", response_class=HTMLResponse)
def comparison_page(request: Request):
    """Serve the model comparison interface."""
    return templates.TemplateResponse("comparison.html", {"request": request})