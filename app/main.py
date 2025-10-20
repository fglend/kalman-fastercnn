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

# ============================================================
# Initialization
# ============================================================
app = FastAPI(title="Faster R-CNN API (Optimized Live Stream)")
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

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
# Live Stream Web Page
# ============================================================
@app.get("/live", response_class=HTMLResponse)
def live_page():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>📡 Faster R-CNN Live Detection</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                background: #0b0b0b;
                color: #fff;
                font-family: 'Segoe UI', Arial, sans-serif;
                text-align: center;
                padding: 20px;
            }
            h2 {
                color: #e63946;
                margin-bottom: 20px;
                font-size: 24px;
            }
            .container {
                max-width: 680px;
                margin: 0 auto;
            }
            .video-container {
                position: relative;
                width: 100%;
                padding-bottom: 75%;
                background: #1a1a1a;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            }
            video, canvas {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
            }
            video {
                background: #000;
            }
            canvas {
                cursor: crosshair;
                z-index: 10;
            }
            .controls {
                margin-top: 20px;
                display: flex;
                gap: 10px;
                justify-content: center;
                flex-wrap: wrap;
            }
            #cameraSelect {
                background: rgba(230, 57, 70, 0.2);
                color: #fff;
                border: 2px solid #e63946;
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 14px;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            #cameraSelect:hover {
                background: rgba(230, 57, 70, 0.3);
            }
            #cameraSelect option {
                background: #1a1a1a;
                color: #fff;
            }
            button {
                background: linear-gradient(135deg, #e63946, #d62828);
                color: #fff;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 10px rgba(230, 57, 70, 0.3);
            }
            button:hover {
                background: linear-gradient(135deg, #d62828, #a71228);
                transform: translateY(-2px);
                box-shadow: 0 6px 15px rgba(230, 57, 70, 0.4);
            }
            button:active {
                transform: translateY(0);
            }
            .info-panel {
                margin-top: 20px;
                background: rgba(255,255,255,0.08);
                border-left: 4px solid #e63946;
                border-radius: 8px;
                padding: 15px;
            }
            #count {
                color: #4caf50;
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 8px;
            }
            #status {
                font-size: 14px;
                color: #aaa;
                margin-bottom: 10px;
            }
            #detections {
                background: rgba(0,0,0,0.3);
                border-radius: 6px;
                padding: 12px;
                text-align: left;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                color: #4caf50;
                max-height: 200px;
                overflow-y: auto;
                line-height: 1.6;
            }
            #detections::-webkit-scrollbar {
                width: 6px;
            }
            #detections::-webkit-scrollbar-track {
                background: rgba(255,255,255,0.05);
                border-radius: 3px;
            }
            #detections::-webkit-scrollbar-thumb {
                background: rgba(230, 57, 70, 0.5);
                border-radius: 3px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>📸 Faster R-CNN Live Detection</h2>
            <div class="video-container">
                <video id="camera" autoplay playsinline muted></video>
                <canvas id="overlay"></canvas>
            </div>
            <div class="controls">
                <select id="cameraSelect">
                    <option value="">📷 Loading cameras...</option>
                </select>
                <button id="toggle">▶ Start Detection</button>
            </div>
            <div class="info-panel">
                <p id="count">Objects detected: 0</p>
                <p id="status">⏳ Waiting...</p>
                <div id="detections">Ready for detections...</div>
            </div>
        </div>

        <script>
            const video = document.getElementById('camera');
            const canvas = document.getElementById('overlay');
            const ctx = canvas.getContext('2d');
            const toggle = document.getElementById('toggle');
            const cameraSelect = document.getElementById('cameraSelect');
            const countText = document.getElementById('count');
            const statusText = document.getElementById('status');
            const detBox = document.getElementById('detections');
            
            let streaming = false, intervalId;
            let videoWidth, videoHeight;
            let currentStream = null;
            let availableCameras = [];

            async function enumerateCameras() {
                try {
                    const devices = await navigator.mediaDevices.enumerateDevices();
                    availableCameras = devices.filter(device => device.kind === 'videoinput');
                    
                    cameraSelect.innerHTML = '';
                    
                    if (availableCameras.length === 0) {
                        cameraSelect.innerHTML = '<option value="">No cameras found</option>';
                        return;
                    }
                    
                    availableCameras.forEach((camera, index) => {
                        const option = document.createElement('option');
                        option.value = camera.deviceId;
                        
                        let label = camera.label || `Camera ${index + 1}`;
                        if (label.toLowerCase().includes('back') || label.toLowerCase().includes('rear')) {
                            label += ' (Rear)';
                            if (index === 0) option.selected = true;
                        } else if (label.toLowerCase().includes('front')) {
                            label += ' (Front)';
                        }
                        
                        option.textContent = label;
                        cameraSelect.appendChild(option);
                    });
                    
                    // Select rear camera by default
                    const rearCamera = availableCameras.find(cam => 
                        cam.label.toLowerCase().includes('back') || 
                        cam.label.toLowerCase().includes('rear')
                    );
                    if (rearCamera) {
                        cameraSelect.value = rearCamera.deviceId;
                    }
                } catch (e) {
                    console.error('Error enumerating cameras:', e);
                    cameraSelect.innerHTML = '<option value="">Error accessing cameras</option>';
                }
            }

            async function setupCamera(deviceId = null) {
                try {
                    if (currentStream) {
                        currentStream.getTracks().forEach(track => track.stop());
                    }
                    
                    const constraints = { 
                        video: { 
                            width: { ideal: 1280 },
                            height: { ideal: 720 }
                        } 
                    };
                    
                    if (deviceId) {
                        constraints.video.deviceId = { exact: deviceId };
                    } else {
                        constraints.video.facingMode = { ideal: 'environment' };
                    }
                    
                    currentStream = await navigator.mediaDevices.getUserMedia(constraints);
                    video.srcObject = currentStream;
                    
                    await new Promise(resolve => {
                        video.onloadedmetadata = () => {
                            videoWidth = video.videoWidth;
                            videoHeight = video.videoHeight;
                            canvas.width = videoWidth;
                            canvas.height = videoHeight;
                            
                            const track = currentStream.getVideoTracks()[0];
                            const settings = track.getSettings();
                            console.log(`Camera: ${settings.facingMode}, Resolution: ${videoWidth}x${videoHeight}`);
                            
                            resolve();
                        };
                    });
                    
                    statusText.textContent = "✅ Camera ready";
                } catch (e) {
                    statusText.textContent = "❌ Camera access denied";
                    console.error('Camera setup error:', e);
                    alert("Cannot access camera: " + e.message);
                }
            }

            cameraSelect.addEventListener('change', (e) => {
                if (streaming) {
                    alert("Stop detection first to switch cameras");
                    return;
                }
                setupCamera(e.target.value);
            });

            async function captureFrame() {
                try {
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = videoWidth;
                    tempCanvas.height = videoHeight;
                    const tempCtx = tempCanvas.getContext('2d');
                    tempCtx.drawImage(video, 0, 0);
                    
                    tempCanvas.toBlob(async (blob) => {
                        const formData = new FormData();
                        formData.append('file', blob, 'frame.jpg');
                        
                        try {
                            const res = await fetch('/predict-image', { 
                                method: 'POST', 
                                body: formData 
                            });
                            const data = await res.json();
                            
                            drawDetections(data.detections || []);
                            countText.textContent = `Objects detected: ${data.num_detections || 0}`;
                            statusText.textContent = "🟢 Detection OK";
                            
                            if (data.detections && data.detections.length > 0) {
                                detBox.innerHTML = data.detections
                                    .map((d, i) => `#${i+1} ID:${d.label_id} | Score:${(d.score).toFixed(2)} | [${Math.round(d.x_min)},${Math.round(d.y_min)},${Math.round(d.x_max)},${Math.round(d.y_max)}]`)
                                    .join('<br>');
                            } else {
                                detBox.innerHTML = "🔍 No detections";
                            }
                        } catch (e) {
                            console.error(e);
                            statusText.textContent = "⚠️ Detection error";
                        }
                    }, 'image/jpeg', 0.85);
                } catch (e) {
                    console.error('Frame capture error:', e);
                    statusText.textContent = "⚠️ Frame capture failed";
                }
            }

            function drawDetections(detections) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#FF0000';
                ctx.font = 'bold 14px Arial';
                
                detections.forEach((d, idx) => {
                    const { x_min, y_min, x_max, y_max, score, label_id } = d;
                    const w = x_max - x_min;
                    const h = y_max - y_min;
                    
                    // Draw rectangle
                    ctx.strokeRect(x_min, y_min, w, h);
                    
                    // Draw label background
                    const label = `ID:${label_id} (${score.toFixed(2)})`;
                    const textMetrics = ctx.measureText(label);
                    const textWidth = textMetrics.width + 8;
                    const textHeight = 20;
                    
                    ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
                    ctx.fillRect(x_min, Math.max(0, y_min - textHeight), textWidth, textHeight);
                    
                    // Draw text
                    ctx.fillStyle = '#FFFF00';
                    ctx.fillText(label, x_min + 4, Math.max(14, y_min - 4));
                });
            }

            toggle.onclick = () => {
                if (!streaming) {
                    streaming = true;
                    toggle.textContent = "⏸ Stop Detection";
                    statusText.textContent = "🟡 Detecting...";
                    intervalId = setInterval(captureFrame, 800);
                } else {
                    streaming = false;
                    toggle.textContent = "▶ Start Detection";
                    clearInterval(intervalId);
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    statusText.textContent = "⏳ Stopped";
                }
            };
            
            setupCamera();
            enumerateCameras();
        </script>
    </body>
    </html>
    """

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