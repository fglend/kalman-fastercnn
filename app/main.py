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
# Predict Video/Image Endpoint
# ============================================================
@app.get("/", response_class=HTMLResponse)
def predict_classes():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>📊 Detection Analysis</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                background: linear-gradient(135deg, #0b0b0b, #1a1a2e);
                color: #fff;
                font-family: 'Segoe UI', Arial, sans-serif;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                color: #e63946;
                margin-bottom: 30px;
                font-size: 28px;
            }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 30px;
                flex-wrap: wrap;
                justify-content: center;
            }
            .tab-btn {
                background: rgba(230, 57, 70, 0.2);
                color: #fff;
                border: 2px solid #e63946;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .tab-btn.active {
                background: #e63946;
                box-shadow: 0 4px 15px rgba(230, 57, 70, 0.4);
            }
            .tab-btn:hover {
                transform: translateY(-2px);
            }
            .tab-content {
                display: none;
                background: rgba(255,255,255,0.05);
                border-radius: 12px;
                padding: 30px;
                backdrop-filter: blur(10px);
            }
            .tab-content.active {
                display: block;
            }
            .upload-section {
                background: rgba(0,0,0,0.3);
                border: 2px dashed #e63946;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin-bottom: 20px;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .upload-section:hover {
                background: rgba(230, 57, 70, 0.1);
                border-color: #ff6b7a;
            }
            .upload-section input {
                display: none;
            }
            .upload-label {
                cursor: pointer;
                font-size: 18px;
                color: #aaa;
            }
            .upload-section.dragging {
                background: rgba(230, 57, 70, 0.2);
                border-color: #ff6b7a;
            }
            .file-info {
                margin-top: 10px;
                font-size: 14px;
                color: #4caf50;
            }
            .controls {
                display: flex;
                gap: 10px;
                justify-content: center;
                flex-wrap: wrap;
                margin-bottom: 20px;
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
            }
            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .results {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 20px;
            }
            @media (max-width: 768px) {
                .results {
                    grid-template-columns: 1fr;
                }
            }
            .result-box {
                background: rgba(0,0,0,0.3);
                border-radius: 10px;
                padding: 20px;
                border-left: 4px solid #e63946;
            }
            .result-box h3 {
                color: #e63946;
                margin-bottom: 15px;
            }
            .result-image {
                width: 100%;
                max-height: 500px;
                border-radius: 8px;
                object-fit: contain;
                margin-bottom: 10px;
            }
            .result-info {
                background: rgba(255,255,255,0.05);
                border-radius: 6px;
                padding: 15px;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                max-height: 300px;
                overflow-y: auto;
                line-height: 1.6;
                color: #4caf50;
            }
            .result-info h4 {
                color: #e63946;
                margin-bottom: 10px;
                margin-top: 10px;
            }
            .result-info h4:first-child {
                margin-top: 0;
            }
            .loading {
                display: none;
                text-align: center;
                padding: 40px;
            }
            .spinner {
                border: 4px solid rgba(230, 57, 70, 0.2);
                border-top: 4px solid #e63946;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .error {
                background: rgba(255, 100, 100, 0.2);
                border-left: 4px solid #ff6b7a;
                padding: 15px;
                border-radius: 8px;
                color: #ff9999;
                margin-top: 10px;
            }
            .success {
                background: rgba(76, 175, 80, 0.2);
                border-left: 4px solid #4caf50;
                padding: 15px;
                border-radius: 8px;
                color: #90ee90;
                margin-top: 10px;
            }
            .download-btn {
                background: #4caf50;
                margin-top: 10px;
            }
            .download-btn:hover {
                background: #45a049;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .stat-card {
                background: rgba(76, 175, 80, 0.1);
                border-left: 4px solid #4caf50;
                padding: 15px;
                border-radius: 6px;
                text-align: center;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #4caf50;
            }
            .stat-label {
                font-size: 12px;
                color: #aaa;
                margin-top: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Faster R-CNN Detection Analysis</h1>
            
            <div class="tabs">
                <button class="tab-btn active" data-tab="image">📷 Image Analysis</button>
                <button class="tab-btn" data-tab="video">🎬 Video Analysis</button>
            </div>

            <!-- IMAGE TAB -->
            <div id="image" class="tab-content active">
                <div class="upload-section" id="imageUpload">
                    <input type="file" id="imageInput" accept="image/*">
                    <label for="imageInput" class="upload-label">
                        Click to upload or drag & drop<br>
                        <span style="font-size: 12px; color: #666;">Supported: JPG, PNG, GIF, WebP</span>
                    </label>
                    <div class="file-info" id="imageFileName"></div>
                </div>

                <div class="controls">
                    <button id="analyzeImageBtn" disabled>Analyze Image</button>
                    <button id="clearImageBtn">Clear</button>
                </div>

                <div class="loading" id="imageLoading">
                    <div class="spinner"></div>
                    <p>Analyzing image...</p>
                </div>

                <div class="results" id="imageResults" style="display: none;">
                    <div class="result-box">
                        <h3>Original Image</h3>
                        <img id="originalImage" class="result-image" alt="Original">
                    </div>
                    <div class="result-box">
                        <h3>Detections Info</h3>
                        <div class="stats" id="imageStats"></div>
                        <div class="result-info" id="imageInfo"></div>
                    </div>
                    <div class="result-box">
                        <h3>Detected Objects</h3>
                        <img id="visualizedImage" class="result-image" alt="Visualized">
                    </div>
                </div>

                <div id="imageError"></div>
            </div>

            <!-- VIDEO TAB -->
            <div id="video" class="tab-content">
                <div class="upload-section" id="videoUpload">
                    <input type="file" id="videoInput" accept="video/*">
                    <label for="videoInput" class="upload-label">
                        Click to upload or drag & drop<br>
                        <span style="font-size: 12px; color: #666;">Supported: MP4, AVI, MOV, MKV</span>
                    </label>
                    <div class="file-info" id="videoFileName"></div>
                </div>

                <div class="controls">
                    <button id="analyzeVideoBtn" disabled>Analyze Video</button>
                    <button id="clearVideoBtn">Clear</button>
                </div>

                <div class="loading" id="videoLoading">
                    <div class="spinner"></div>
                    <p>Processing video... This may take a while</p>
                </div>

                <div id="videoResult" style="display: none;">
                    <div class="success" id="videoSuccess"></div>
                    <button class="download-btn" id="downloadVideoBtn">Download Analyzed Video</button>
                </div>

                <div id="videoError"></div>
            </div>
        </div>

        <script>
            // Tab switching
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                    e.target.classList.add('active');
                    document.getElementById(e.target.dataset.tab).classList.add('active');
                });
            });

            // ============ IMAGE HANDLING ============
            const imageUpload = document.getElementById('imageUpload');
            const imageInput = document.getElementById('imageInput');
            const imageFileName = document.getElementById('imageFileName');
            const analyzeImageBtn = document.getElementById('analyzeImageBtn');
            const clearImageBtn = document.getElementById('clearImageBtn');
            const imageLoading = document.getElementById('imageLoading');
            const imageResults = document.getElementById('imageResults');
            const imageError = document.getElementById('imageError');
            let selectedImage = null;

            // Drag and drop for image
            imageUpload.addEventListener('dragover', (e) => {
                e.preventDefault();
                imageUpload.classList.add('dragging');
            });
            imageUpload.addEventListener('dragleave', () => {
                imageUpload.classList.remove('dragging');
            });
            imageUpload.addEventListener('drop', (e) => {
                e.preventDefault();
                imageUpload.classList.remove('dragging');
                const files = e.dataTransfer.files;
                if (files.length) imageInput.files = files;
                handleImageSelect();
            });

            imageInput.addEventListener('change', handleImageSelect);

            function handleImageSelect() {
                const file = imageInput.files[0];
                if (file) {
                    selectedImage = file;
                    imageFileName.textContent = `Selected: ${file.name}`;
                    analyzeImageBtn.disabled = false;
                    imageError.innerHTML = '';
                }
            }

            analyzeImageBtn.addEventListener('click', analyzeImage);
            clearImageBtn.addEventListener('click', () => {
                imageInput.value = '';
                imageFileName.textContent = '';
                imageResults.style.display = 'none';
                imageLoading.style.display = 'none';
                analyzeImageBtn.disabled = true;
                selectedImage = null;
                imageError.innerHTML = '';
            });

            async function analyzeImage() {
                if (!selectedImage) return;

                imageLoading.style.display = 'block';
                imageResults.style.display = 'none';
                imageError.innerHTML = '';

                const formData = new FormData();
                formData.append('file', selectedImage);

                try {
                    // Get predictions
                    const predRes = await fetch('/predict-image', { method: 'POST', body: formData });
                    const predData = await predRes.json();

                    // Get visualized image
                    const visRes = await fetch('/visualize-image', { method: 'POST', body: formData });
                    const visBlob = await visRes.blob();

                    // Display original
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        document.getElementById('originalImage').src = e.target.result;
                    };
                    reader.readAsDataURL(selectedImage);

                    // Display visualized
                    const visUrl = URL.createObjectURL(visBlob);
                    document.getElementById('visualizedImage').src = visUrl;

                    // Display stats and info
                    const stats = document.getElementById('imageStats');
                    stats.innerHTML = `
                        <div class="stat-card">
                            <div class="stat-value">${predData.num_detections}</div>
                            <div class="stat-label">Objects Detected</div>
                        </div>
                    `;

                    const info = document.getElementById('imageInfo');
                    if (predData.detections.length > 0) {
                        info.innerHTML = '<h4>Detection Results:</h4>' + predData.detections
                            .map((d, i) => `<div>#${i+1} | ID:${d.label_id} | Score:${d.score.toFixed(4)} | Box:[${Math.round(d.x_min)}, ${Math.round(d.y_min)}, ${Math.round(d.x_max)}, ${Math.round(d.y_max)}]</div>`)
                            .join('');
                    } else {
                        info.innerHTML = '<h4>No objects detected</h4>';
                    }

                    imageLoading.style.display = 'none';
                    imageResults.style.display = 'grid';
                } catch (e) {
                    imageLoading.style.display = 'none';
                    imageError.innerHTML = `<div class="error">Error: ${e.message}</div>`;
                    console.error(e);
                }
            }

            // ============ VIDEO HANDLING ============
            const videoUpload = document.getElementById('videoUpload');
            const videoInput = document.getElementById('videoInput');
            const videoFileName = document.getElementById('videoFileName');
            const analyzeVideoBtn = document.getElementById('analyzeVideoBtn');
            const clearVideoBtn = document.getElementById('clearVideoBtn');
            const videoLoading = document.getElementById('videoLoading');
            const videoResult = document.getElementById('videoResult');
            const videoError = document.getElementById('videoError');
            let selectedVideo = null;

            // Drag and drop for video
            videoUpload.addEventListener('dragover', (e) => {
                e.preventDefault();
                videoUpload.classList.add('dragging');
            });
            videoUpload.addEventListener('dragleave', () => {
                videoUpload.classList.remove('dragging');
            });
            videoUpload.addEventListener('drop', (e) => {
                e.preventDefault();
                videoUpload.classList.remove('dragging');
                const files = e.dataTransfer.files;
                if (files.length) videoInput.files = files;
                handleVideoSelect();
            });

            videoInput.addEventListener('change', handleVideoSelect);

            function handleVideoSelect() {
                const file = videoInput.files[0];
                if (file) {
                    selectedVideo = file;
                    videoFileName.textContent = `Selected: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
                    analyzeVideoBtn.disabled = false;
                    videoError.innerHTML = '';
                }
            }

            analyzeVideoBtn.addEventListener('click', analyzeVideo);
            clearVideoBtn.addEventListener('click', () => {
                videoInput.value = '';
                videoFileName.textContent = '';
                videoResult.style.display = 'none';
                videoLoading.style.display = 'none';
                analyzeVideoBtn.disabled = true;
                selectedVideo = null;
                videoError.innerHTML = '';
            });

            async function analyzeVideo() {
                if (!selectedVideo) return;

                videoLoading.style.display = 'block';
                videoResult.style.display = 'none';
                videoError.innerHTML = '';

                const formData = new FormData();
                formData.append('file', selectedVideo);

                try {
                    const res = await fetch('/analyze-video', { method: 'POST', body: formData });
                    
                    if (!res.ok) {
                        const errData = await res.json();
                        throw new Error(errData.error || 'Video analysis failed');
                    }

                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);

                    videoLoading.style.display = 'none';
                    document.getElementById('videoSuccess').textContent = '✅ Video analyzed successfully!';
                    document.getElementById('downloadVideoBtn').onclick = () => {
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'analyzed_' + selectedVideo.name;
                        a.click();
                    };
                    videoResult.style.display = 'block';
                } catch (e) {
                    videoLoading.style.display = 'none';
                    videoError.innerHTML = `<div class="error">Error: ${e.message}</div>`;
                    console.error(e);
                }
            }
        </script>
    </body>
    </html>
    """

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
                background: #000;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.5);
                max-width: 100%;
            }
            video {
                display: block;
                width: 100%;
                height: auto;
                background: #000;
            }
            canvas {
                position: absolute;
                top: 0;
                left: 0;
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
                            
                            // Set canvas to video dimensions
                            canvas.width = videoWidth;
                            canvas.height = videoHeight;
                            
                            // Position canvas to match video display
                            const rect = video.getBoundingClientRect();
                            canvas.style.width = rect.width + 'px';
                            canvas.style.height = rect.height + 'px';
                            
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
                
                // Fill entire canvas with semi-transparent overlay to test alignment
                ctx.fillStyle = 'rgba(0, 255, 0, 0.15)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Get display size vs actual size for scaling
                const displayWidth = canvas.offsetWidth;
                const displayHeight = canvas.offsetHeight;
                const scaleX = displayWidth / canvas.width;
                const scaleY = displayHeight / canvas.height;
                
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#FF0000';
                ctx.font = 'bold 14px Arial';
                
                detections.forEach((d, idx) => {
                    let { x_min, y_min, x_max, y_max, score, label_id } = d;
                    
                    // Scale coordinates
                    x_min *= scaleX;
                    y_min *= scaleY;
                    x_max *= scaleX;
                    y_max *= scaleY;
                    
                    const w = x_max - x_min;
                    const h = y_max - y_min;
                    
                    // Draw filled rectangle with opacity
                    ctx.fillStyle = 'rgba(255, 0, 0, 0.15)';
                    ctx.fillRect(x_min, y_min, w, h);
                    
                    // Draw rectangle border
                    ctx.strokeStyle = '#FF0000';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x_min, y_min, w, h);
                    
                    // Draw label background
                    const label = `ID:${label_id} (${score.toFixed(2)})`;
                    const textMetrics = ctx.measureText(label);
                    const textWidth = textMetrics.width + 8;
                    const textHeight = 20;
                    
                    ctx.fillStyle = 'rgba(255, 0, 0, 0.8)';
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