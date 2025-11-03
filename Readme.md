# 🚀 Faster R-CNN w/ Kalman Filter API (FastAPI + PyTorch)

A high-performance web API for **object detection, video analytics, and evaluation** using a trained **Faster R-CNN with Kalman Filter** model.  
Includes endpoints for **image prediction**, **video analysis**, **live streaming**, and **automatic gallery management** — optimized for both **CPU** and **GPU** inference.

---

## 🧠 Features

- ⚡ **FastAPI REST API** with threaded inference
- 🖼️ Image & video upload endpoints for real-time detection
- 🎯 **Kalman Filter-based tracking** for smoother temporal consistency
- 📊 **Automatic saving and gallery view** for all analyzed media
- 📁 Organized storage:
  - `/results/images` — original uploads
  - `/results/json` — raw detections
  - `/results/coco` — COCO-format annotations
  - `/results/videos` — processed clips
  - `/results/videos_json` — video metadata
- 🧩 Built-in web UI with:
  - **📷 Image analysis**
  - **🎬 Video analysis**
  - **🖼️ Gallery viewer with download buttons**
- 🧮 Compatible with **CPU**, **CUDA**, and **Apple Silicon (MPS)**
- 🐳 Fully **Docker-ready**, with volume support for persistent results

---

## 🧠 Trained Model

📦 **Faster R-CNN + Kalman Filter weights**  
🔗 [Request Access Here](https://drive.google.com/file/d/1KC9LZ1u8av3O4lO-_VJ8r9P_2PHnzsLU/view?usp=drive_link)

---

## 🖼️ System UI Samples

Hosted via [LocalTunnel](https://theboroer.github.io/localtunnel-www/)

### 🔹 Image / Video Analysis

🔗 [https://gd-live.loca.lt/](https://gd-live.loca.lt/)

<p align="center"><img src="/assets/1.png" width="600"/></p>

### 🔹 Live Detection

🔗 [https://gd-live.loca.lt/live](https://gd-live.loca.lt/live)

<p align="center"><img src="/assets/2.png" width="600"/></p>

### 🔹 API Docs

🔗 [https://gd-live.loca.lt/docs](https://gd-live.loca.lt/docs)

<p align="center"><img src="/assets/3.png" width="600"/></p>

---

## 📦 Requirements

Python 3.10+

### Dependencies

```bash
fastapi==0.115.0
uvicorn[standard]==0.30.6
torch==2.4.1
torchvision==0.19.1
pillow==10.4.0
pydantic==1.10.13
numpy==1.26.4
python-multipart==0.0.9
opencv-python==4.10.0.84
```

---

## ⚙️ Installation (Local)

```bash
# 1️⃣ Clone the repo
git clone https://github.com/fglend/kalman-fastercnn.git
cd kalman-fastercnn

# 2️⃣ Create a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows PowerShell
# or
source venv/bin/activate    # macOS / Linux

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the API locally
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Then open:

- Docs → [http://localhost:8080/docs](http://localhost:8080/docs)
- Live → [http://localhost:8080/live](http://localhost:8080/live)

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t fasterrcnn-api .
```

### Run the container

```bash
docker run -p 8080:8080 fasterrcnn-api
```

### With mounted volumes

```bash
# macOS / Linux
docker run -p 8080:8080 -v $(pwd)/models:/models -v $(pwd)/results:/results fasterrcnn-api

# Windows PowerShell
docker run -p 8080:8080 -v ${PWD}/models:/models -v ${PWD}/results:/results fasterrcnn-api
```

---

## 📂 Auto-Saving & Gallery System

Each `/predict-image` or `/visualize-image` request automatically saves:

- Original input → `/results/images`
- JSON detections → `/results/json`
- COCO annotation → `/results/coco`

Videos analyzed via `/start-analyze` are stored under `/results/videos` with metadata in `/results/videos_json`.

### Example Output

```
✅ /results/images/20251103_125959.jpg
✅ /results/json/20251103_125959.json
✅ /results/coco/20251103_125959.json
```

---

## 🔍 Key API Endpoints

### **Health Check**

**GET** `/health`

> Returns device info and runtime status.

---

### **Image Prediction**

**POST** `/predict-image`  
Returns detection boxes and scores as JSON.

```bash
curl -X POST "http://localhost:8080/predict-image" -F "file=@sample.jpg"
```

---

### **Visualization**

**POST** `/visualize-image`  
Returns grayscale, darkened annotated image with red bounding boxes.

```bash
curl -X POST "http://localhost:8080/visualize-image" -F "file=@sample.jpg" --output output.jpg
```

---

### **Video Analysis**

**POST** `/start-analyze`  
Processes uploaded video asynchronously with progress tracking.

Progress can be checked with:

- `/progress/{job_id}` — current status
- `/result/{job_id}` — final MP4 result stream

---

### **Gallery Endpoints**

| Type          | Endpoint                         | Description                    |
| ------------- | -------------------------------- | ------------------------------ |
| 📋 List       | `/gallery/list`                  | Returns all saved detections   |
| 🖼️ Image      | `/gallery/image/{timestamp}`     | Original saved image           |
| 🟥 Visualized | `/gallery/visualize/{timestamp}` | Annotated image with red boxes |
| 📄 JSON       | `/gallery/json/{timestamp}`      | Detection data                 |
| ❌ Delete     | `/gallery/delete/{timestamp}`    | Remove saved item              |

Video gallery:
| Type | Endpoint | Description |
|------|-----------|-------------|
| 🎞️ List | `/gallery/videos/list` | List analyzed videos |
| ▶️ Stream | `/gallery/videos/stream/{timestamp}` | Stream processed video |
| 🧾 Metadata | `/gallery/videos/json/{timestamp}` | Video metadata |
| 🗑️ Delete | `/gallery/videos/delete/{timestamp}` | Remove video files |

---

## 🧩 Web UI Tabs

| Tab                   | Description                                                                               |
| --------------------- | ----------------------------------------------------------------------------------------- |
| **📷 Image Analysis** | Upload or capture photo for detection                                                     |
| **🎬 Video Analysis** | Upload MP4 for frame-by-frame inference                                                   |
| **🖼️ Gallery**        | View all saved detections with modal viewer                                               |
|                       | ➕ Each modal includes “📷 Download Original” & “📥 Download with Bounding Boxes” buttons |

---

## ⚙️ Configuration

Edit `app/config.py`:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NUM_THREADS = 4
MODEL_PATH = "models/best_model.pth"
SCORE_THRESH = 0.5
```

---

## ⚡ Performance Tips

- 🔥 Use **CUDA GPU** or **Apple MPS** for max speed
- 🖼️ Adjust image `T.Resize()` to 384×384 for higher FPS
- 🎥 Modify `skip_frames` in `process_video_job()` for speed/accuracy balance
- 💡 Store `/results` on SSD or mounted Docker volume for better I/O

---

## 📁 Project Layout

```
.
├── app/
│   ├── main.py              # FastAPI + endpoints
│   ├── model.py             # Model loader
│   ├── predict_utils.py     # Preprocessing helpers
│   ├── config.py            # Settings
│   └── templates/           # HTML frontend
├── models/
│   └── best_model.pth
├── results/
│   ├── images/
│   ├── json/
│   ├── coco/
│   ├── videos/
│   └── videos_json/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 👨‍💻 Author

**Glend Dale Ferrer**  
📧 mgdferrer@tip.edu.ph

---

## 📜 License

MIT License © 2025 Glend Dale Ferrer
