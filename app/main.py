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

    # Start background inference thread
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
# Visualize Image Endpoint
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
        draw.text((x_min, max(y_min - 10, 0)),
                  f"ID:{int(l)} | {s:.2f}", fill="yellow", font=font)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return StreamingResponse(img_bytes, media_type="image/jpeg")

# ============================================================
# Live Stream Web Page
# ============================================================
@app.get("/live", response_class=HTMLResponse)
def live_page():
    """Enhanced live detection UI (works on laptop and mobile)."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>📡 Faster R-CNN Live Detection</title>
        <style>
            body {
                background: #0b0b0b;
                color: #fff;
                font-family: Arial, sans-serif;
                text-align: center;
                margin: 0;
                padding: 0;
            }
            h2 {
                color: #e63946;
                margin-top: 15px;
            }
            video, canvas {
                border-radius: 10px;
                width: 90%;
                max-width: 640px;
                margin-top: 15px;
            }
            button {
                background: #e63946;
                color: #fff;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                margin-top: 10px;
                font-size: 16px;
                cursor: pointer;
            }
            #count {
                color: #4caf50;
                margin-top: 8px;
                font-size: 18px;
            }
            #status {
                font-size: 14px;
                color: #ccc;
            }
            #detections {
                margin-top: 10px;
                background: rgba(255,255,255,0.05);
                padding: 10px;
                border-radius: 8px;
                width: 85%;
                max-width: 640px;
                margin-left: auto;
                margin-right: auto;
                text-align: left;
                font-family: monospace;
                font-size: 13px;
                white-space: pre-line;
            }
        </style>
    </head>
    <body>
        <h2>📸 Faster R-CNN Live Detection</h2>
        <video id="camera" autoplay playsinline muted></video>
        <canvas id="overlay"></canvas><br>
        <button id="toggle">Start Detection</button>
        <p id="count">Colonies detected: 0</p>
        <p id="status"></p>
        <div id="detections"></div>

        <script>
            const video=document.getElementById('camera');
            const canvas=document.getElementById('overlay');
            const ctx=canvas.getContext('2d');
            const toggle=document.getElementById('toggle');
            const countText=document.getElementById('count');
            const statusText=document.getElementById('status');
            const detBox=document.getElementById('detections');
            let streaming=false, intervalId;

            async function setupCamera(){
                try{
                    const stream=await navigator.mediaDevices.getUserMedia({video:true});
                    video.srcObject=stream;
                    await new Promise(r=>video.onloadedmetadata=r);
                    canvas.width=video.videoWidth;
                    canvas.height=video.videoHeight;
                    statusText.textContent="✅ Camera ready";
                }catch(e){alert("❌ Cannot access camera: "+e);}
            }

            async function captureFrame(){
                const c=document.createElement('canvas');
                c.width=video.videoWidth;
                c.height=video.videoHeight;
                c.getContext('2d').drawImage(video,0,0);
                const blob=await new Promise(r=>c.toBlob(r,'image/jpeg'));
                const formData=new FormData(); formData.append('file',blob,'frame.jpg');
                try{
                    const res=await fetch('/predict-image',{method:'POST',body:formData});
                    const data=await res.json();
                    drawDetections(data.detections);
                    countText.textContent=`Objects detected: ${data.num_detections}`;
                    statusText.textContent="🟢 Detection OK";
                    
                    // Show detection info
                    if (data.detections.length > 0) {
                        detBox.innerHTML = data.detections
                            .map((d,i)=>`#${i+1} → ID:${d.label_id} | Score:${d.score.toFixed(2)} | Box:[${d.x_min.toFixed(0)}, ${d.y_min.toFixed(0)}, ${d.x_max.toFixed(0)}, ${d.y_max.toFixed(0)}]`)
                            .join('\\n');
                    } else {
                        detBox.innerHTML = "No detections.";
                    }

                }catch(e){
                    console.error(e);
                    statusText.textContent="⚠️ "+e.message;
                }
            }

            function drawDetections(detections){
                ctx.clearRect(0,0,canvas.width,canvas.height);
                detections.forEach(d=>{
                    const{x_min,y_min,x_max,y_max,score,label_id}=d;
                    ctx.strokeStyle="red"; ctx.lineWidth=2;
                    ctx.strokeRect(x_min,y_min,x_max-x_min,y_max-y_min);
                    ctx.fillStyle="yellow"; ctx.font="14px Arial";
                    ctx.fillText(`ID:${label_id} (${score.toFixed(2)})`,x_min+4,y_min-6);
                });
            }

            toggle.onclick=()=>{
                if(!streaming){
                    streaming=true;
                    toggle.textContent="Stop Detection";
                    intervalId=setInterval(captureFrame,800);
                }else{
                    streaming=false;
                    toggle.textContent="Start Detection";
                    clearInterval(intervalId);
                    ctx.clearRect(0,0,canvas.width,canvas.height);
                    detBox.innerHTML="";
                }
            };
            setupCamera();
        </script>
    </body>
    </html>
    """

# ============================================================
# Optimized Video Feed (for debug mode)
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
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/video-feed")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")