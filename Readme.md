# 🚀 Faster R-CNN w/ Kalman Filter API (FastAPI + PyTorch)

A high-performance web API for **object detection, video analytics, live streaming, and gallery management** using trained **Faster R-CNN (Stand-alone model & w/ Kalman Filter)** checkpoints.  
Includes endpoints for **single-image prediction**, **video analysis** (sync & async), **live MJPEG stream**, **model comparison**, and **automatic COCO export** — optimized for **CPU**, **CUDA**, and **Apple Silicon (MPS)**.

---

## 🧠 What’s inside

- ⚡ **FastAPI REST API** with threaded live inference (MJPEG stream)
- 🧰 **Two models** loaded from `.pth`: **Stand-Alone** and **Hybrid** (choose per endpoint)
- 🎯 Single-image **JSON prediction** + **on-the-fly visualization**
- 🎬 Video analysis:
  - **Async pipeline**: `/start-analyze` → `/progress/{job_id}` → `/result/{job_id}` (+ saved to gallery)
  - **Sync pipeline**: `/analyze-video` (returns processed MP4 immediately)
- 🗂️ **Auto-saving** of originals, detections, and **COCO** annotations
- 🖼️ **Image & video galleries** (with thumbnails, metadata, and deletion)
- 🆚 **Model comparison** (side-by-side image visualization + JSON diffs)
- 🌐 **Built-in web UI** pages (`/`, `/live`, `/comparison`) and OpenAPI docs (`/docs`)
- 🐳 **Docker-ready** (mount `models/` and `results/` for persistence)
- 🔒 CORS pre-configured for:
  - `https://gd-live.com`, `http://localhost:8080`, `http://localhost:3000`

**Storage layout**

- `/results/images` — original uploaded images  
- `/results/json` — raw detections per image  
- `/results/coco` — COCO-format annotations per image  
- `/results/videos` — processed MP4 clips  
- `/results/videos_json` — per-video metadata  
- `/results/comparisons` — comparison image + JSON

---

## 📦 Trained Models

This API expects two checkpoints:

- `model_1_path` → e.g., `models/sample_model_1.pth`  
- `model_2_path` → e.g., `models/sample_model_2.pth`

(You can use your own weights trained on your dataset.)

---

## 🖼️ System UI Samples

Hosted via [LocalTunnel](https://theboroer.github.io/localtunnel-www/) (example links):

- **Image / Video Analysis**: `https://gd-live.loca.lt/`  
  <p align="center"><img src="/assets/1.png" width="600"/></p>

- **Live Detection**: `https://gd-live.loca.lt/live`  
  <p align="center"><img src="/assets/2.png" width="600"/></p>

- **API Docs**: `https://gd-live.loca.lt/docs`  
  <p align="center"><img src="/assets/3.png" width="600"/></p>

---

## 📦 Requirements

**Python 3.10+**

### Dependencies

```bash
fastapi==0.115.0
uvicorn[standard]==0.30.6
torch==2.4.1
torchvision==0.19.1
pillow==10.4.0
numpy==1.26.4
python-multipart==0.0.9
opencv-python==4.10.0.84
jinja2==3.1.4
pydantic==2.8.2
pydantic-settings==2.5.2
```

> If your `app/config.py` uses Pydantic Settings v1, pin `fastapi<0.110` and `pydantic<2`. Otherwise use the versions above.

---

## ⚙️ Installation (Local)

```bash
# 1️⃣ Clone
git clone https://github.com/fglend/kalman-fastercnn.git
cd kalman-fastercnn

# 2️⃣ Create & activate venv
python -m venv venv
# Windows (PowerShell)
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

# 3️⃣ Install deps
pip install -r requirements.txt

# 4️⃣ (Optional) Set environment variables
# PowerShell
$env:MODEL_1_PATH="models/sample_1.pth"
$env:MODEL_2_PATH="models/sample_2.pth"
$env:RESULTS_DIR="results"
$env:NUM_CLASSES="7"

# bash/zsh
export MODEL_1_PATH=models/sample_1.pth
export MODEL_2_PATH=models/sample_2.pth
export RESULTS_DIR=results
export NUM_CLASSES=7

# 5️⃣ Run the API
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Open:

- Docs → `http://localhost:8080/docs`  
- Live → `http://localhost:8080/live`

---

## 🐳 Docker

```bash
# Build
docker build -t fasterrcnn-kalman-api .

# Run (basic)
docker run -p 8080:8080 fasterrcnn-kalman-api

# Run with mounted volumes (recommended)
# macOS / Linux
docker run -p 8080:8080   -v $(pwd)/models:/models   -v $(pwd)/results:/results   -e MODEL_1_PATH=/models/sample_1.pth   -e MODEL_2_PATH=/models/sample_2.pth   -e RESULTS_DIR=/results   -e NUM_CLASSES=7   fasterrcnn-kalman-api

# Windows PowerShell
docker run -p 8080:8080 `
  -v ${PWD}/models:/models `
  -v ${PWD}/results:/results `
  -e MODEL_1_PATH=/models/sample_1.pth `
  -e MODEL_2_PATH=/models/sample_2.pth `
  -e RESULTS_DIR=/results `
  -e NUM_CLASSES=7 `
 fasterrcnn-kalman-api
```

---

## 🔧 Configuration

`app/config.py` (example)

```python
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NUM_THREADS = 4
SCORE_THRESH = 0.5
```

**Environment variables used in `app.main`:**

- `MODEL_1_PATH` (default: `models/sample_1.pth`)
- `MODEL_2_PATH` (default: `models/sample_2.pth`)
- `RESULTS_DIR` (default: `/results`)
- `NUM_CLASSES` (default: `7`)

CORS is enabled for: `https://gd-live.com`, `http://localhost:8080`, `http://localhost:3000`.

---

## 🔍 Key API Endpoints

### Health

**GET** `/health` → returns runtime info:

```json
{ "status": "ok", "device": "cuda|mps|cpu", "models": ["standalone","hybrid"] }
```

---

### Single-Image Prediction

**POST** `/predict-image` → JSON detections (uses **ResNet-101**)

```bash
curl -X POST "http://localhost:8080/predict-image"   -F "file=@sample.jpg"
```

**Response (example)**

```json
{
  "detections": [
    {"x_min": 12.3, "y_min": 45.6, "x_max": 123.4, "y_max": 234.5, "score": 0.93, "label_id": 3}
  ],
  "num_detections": 1
}
```

Saves **image**, **JSON**, and **COCO** in `/results/...` (done asynchronously).

---

### Single-Image Visualization

**POST** `/visualize-image` → JPEG with **grayscale + darkened** background and **red boxes**

```bash
curl -X POST "http://localhost:8080/visualize-image"   -F "file=@sample.jpg" --output output.jpg
```

---

### Video Analysis — **Async Job**

1) **POST** `/start-analyze` → `{ "job_id": "<uuid>" }`  
2) **GET** `/progress/{job_id}` → `{ "status": "...", "progress": 42.0 }`  
3) **GET** `/result/{job_id}` → MP4 stream (**adds `X-Video-Timestamp` header**)

- Frames are processed with **grayscale + darken**, resize to **384×384** internally for speed, and **FP16** on CUDA.
- Results saved to `/results/videos` + metadata in `/results/videos_json`.

---

### Video Analysis — **Sync**

**POST** `/analyze-video` → returns processed **MP4** immediately (no job tracking)

```bash
curl -X POST "http://localhost:8080/analyze-video"   -F "file=@input.mp4" --output out.mp4
```

---

### Live Stream

- **GET** `/live` → HTML page  
- **GET** `/video-feed` → MJPEG stream (for embedding)

---

### Gallery (Images)

| Type          | Endpoint                              | Description                               |
|---------------|----------------------------------------|-------------------------------------------|
| 📋 List       | `GET /gallery/list`                    | List saved images with counts             |
| 🖼️ Original   | `GET /gallery/image/{timestamp}`       | Stream original uploaded image            |
| 🟥 Visualized | `GET /gallery/visualize/{timestamp}`   | On-the-fly boxes over saved image         |
| 📄 JSON       | `GET /gallery/json/{timestamp}`        | Raw detections                            |
| 🗑️ Delete     | `DELETE /gallery/delete/{timestamp}`   | Remove image/JSON/COCO                    |

---

### Gallery (Videos)

| Type        | Endpoint                                         | Description                                  |
|-------------|---------------------------------------------------|----------------------------------------------|
| 🎞️ List    | `GET /gallery/videos/list`                        | List analyzed videos                         |
| ▶️ Stream   | `GET /gallery/videos/stream/{timestamp}`          | Stream processed video                       |
| 🧾 Metadata | `GET /gallery/videos/json/{timestamp}`            | Video metadata JSON                          |
| 🖼️ Thumb    | `GET /gallery/videos/thumbnail/{timestamp}`       | JPEG thumbnail (first frame)                 |
| 🗑️ Delete   | `DELETE /gallery/videos/delete/{timestamp}`       | Remove video + metadata                      |

---

### Model Comparison

| Type              | Endpoint                    | Description                                   |
|-------------------|-----------------------------|-----------------------------------------------|
| 🧪 JSON compare   | `POST /compare-models`      | Runs **Standalone Model** & **Hybrid Model**, returns both |
| 🖼️ Visual compare | `POST /visualize-comparison`| Side-by-side annotated JPEG      |
| 🖥️ UI page        | `GET /comparison`           | Comparison web interface                      |

Comparison artifacts saved under `/results/comparisons`.

---

## 📂 Project Layout

```
.
├── app/
│   ├── main.py              # FastAPI + endpoints
│   ├── model.py             # Checkpoint loader
│   ├── predict_utils.py     # Preprocessing & filtering
│   ├── config.py            # Settings (DEVICE, NUM_THREADS, SCORE_THRESH)
│   └── templates/           # index.html, live.html, comparison.html
├── models/
│   ├── sample_1.pth
│   └── sample_2.pth
├── results/
│   ├── images/
│   ├── json/
│   ├── coco/
│   ├── videos/
│   ├── videos_json/
│   └── comparisons/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚡ Performance Notes

- GPU: **CUDA** (with **FP16**) or **Apple MPS** recommended
- Image resize: **384×384** (video path) or **512×512** (image path); trade speed vs. accuracy
- Video: `skip_frames=5` by default; lower for higher accuracy, higher for speed
- Model heads tuned at runtime:
  - `roi_heads.detections_per_img = 350`
  - `rpn.pre_nms_top_n_test = 2000`, `rpn.post_nms_top_n_test = 2000`
- Store `/results` on SSD or mounted Docker volume

---

## 👨‍💻 Author

**Glend Dale Ferrer**  
📧 glendferrer05@gmail.com

---

## 📜 License

MIT License © 2025 Glend Dale Ferrer
