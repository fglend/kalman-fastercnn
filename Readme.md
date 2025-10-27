# 🚀 Faster R-CNN w/ Kalman Filter API (FastAPI + PyTorch)

A lightweight web API for **real-time object detection** using a trained **Faster R-CNN with Kalman Filter** model.  
Includes endpoints for image prediction, visualization, and live camera streaming — optimized for smooth performance on both **macOS (M-series)** and **Windows**.

---

## 🧠 Features
- **REST API** built with FastAPI  
- **Image upload** detection endpoint (`/predict-image`)  
- **Visualization endpoint** returning annotated images (`/visualize-image`)  
- **Live camera streaming** (`/live`) with threaded inference  
- **Automatic saving** of:
  - 🖼️ Predicted images (`/results/images`)
  - 📄 Detection JSON (`/results/json`)
  - 📦 COCO-format annotations (`/results/coco`)
- Compatible with **CPU**, **CUDA**, and **Apple Silicon (MPS)**  
- Fully **Docker-ready** and works with or without attached storage volumes  

---

## 🧠 Trained Model
📦 Model weights (Faster R-CNN + Kalman Filter)  
🔗 [Request Access Here](https://drive.google.com/file/d/1KC9LZ1u8av3O4lO-_VJ8r9P_2PHnzsLU/view?usp=drive_link)

---

## 🖼️ System Sample

Hosted via [LocalTunnel](https://theboroer.github.io/localtunnel-www/).  
LocalTunnel uses your **local IP address** as a one-time password (see credentials in the shared Drive `.txt` file).

### 🔹 Image / Video Analysis  
🔗 [https://gd-live.loca.lt/](https://gd-live.loca.lt/)

<p align="center">
  <img src="/assets/1.png" alt="System Sample" width="600"/>
</p>

### 🔹 Live Detection  
🔗 [https://gd-live.loca.lt/live](https://gd-live.loca.lt/live)

<p align="center">
  <img src="/assets/2.png" alt="System Sample" width="600"/>
</p>

### 🔹 API Docs  
🔗 [https://gd-live.loca.lt/docs](https://gd-live.loca.lt/docs)

<p align="center">
  <img src="/assets/3.png" alt="System Sample" width="600"/>
</p>

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

## ⚙️ Installation (Local – Mac / Windows)

```bash
# 1️⃣ Clone the repository
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
- API Docs → [http://localhost:8080/docs](http://localhost:8080/docs)
- Live Stream → [http://localhost:8080/live](http://localhost:8080/live)

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t fasterrcnn-api .
```

### 🧩 Run the Container
```bash
# Simple run
docker run -p 8080:8080 fasterrcnn-api
```

### 💾 Mount your model or results folder
```bash
# macOS / Linux
docker run -p 8080:8080 -v $(pwd)/models:/models -v $(pwd)/results:/results fasterrcnn-api

# Windows PowerShell
docker run -p 8080:8080 -v ${PWD}/models:/models -v ${PWD}/results:/results fasterrcnn-api
```

> The container will automatically detect `/models` and `/results`.  
> If no volume is attached, it uses Docker’s internal storage (ephemeral).

---

## 🧩 Auto-Saving Behavior

Whenever `/predict-image` or `/visualize-image` is called, the app automatically:
1. Saves the **uploaded image** to `/results/images/`
2. Writes a **detection JSON** file to `/results/json/`
3. Generates a **COCO-format annotation** file to `/results/coco/`

### Example:
```
✅ Saved: /results/images/20251027_031154.jpg  
✅ Saved: /results/json/20251027_031154.json  
✅ Saved: /results/coco/20251027_031154.json
```

---

## 🔍 API Endpoints

### **1️⃣ Health Check**
**GET** `/health`  
Returns the model’s current device and runtime info.

---

### **2️⃣ Predict Image**
**POST** `/predict-image`  
Uploads an image and returns detections in JSON format.  
Also saves predictions automatically.

```bash
curl -X POST "http://localhost:8080/predict-image" -F "file=@sample.jpg"
```

---

### **3️⃣ Visualize Image**
**POST** `/visualize-image`  
Returns an annotated image (JPEG) showing all detections.

```bash
curl -X POST "http://localhost:8080/visualize-image" -F "file=@sample.jpg" --output output.jpg
```

---

### **4️⃣ Live Stream**
**GET** `/live`  
Starts live camera detection (works on Chrome desktop or mobile devices on the same network).

---

## ⚙️ Configuration

Edit `app/config.py` for environment and threshold setup:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NUM_THREADS = 4
MODEL_PATH = "models/best_model.pth"
SCORE_THRESH = 0.5
```

---

## ⚡ Performance Tips
- 🧮 Use **CUDA (NVIDIA)** or **MPS (Mac)** for faster inference  
- 🖼️ Lower `T.Resize()` in preprocessing for better FPS  
- ⚙️ Adjust frame skip in `generate_frames()` to balance accuracy vs speed  
- 🚀 Close unused browser tabs while streaming to improve latency  

---

## 🧠 Notes for macOS Users
If you see:
```
[ WARN:0] VIDEOIO(V4L2:/dev/video0): can't open camera by index
```
Try changing:
```python
cap = cv2.VideoCapture(1)
```
Or enable camera permissions:
> **System Settings → Privacy & Security → Camera → Allow Terminal / IDE**

---

## 📁 Project Structure
```
.
├── app/
│   ├── main.py              # FastAPI app
│   ├── model.py             # Model loader
│   ├── predict_utils.py     # Preprocessing & filtering
│   ├── config.py            # Environment configuration
│   └── templates/           # HTML UI templates
├── models/
│   └── best_model.pth       # Trained weights
├── results/
│   ├── images/              # Saved predictions
│   ├── json/                # Detection outputs
│   └── coco/                # COCO annotations
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
