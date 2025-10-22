# 🚀 Faster R-CNN w/ Kalman-Filter API with Live Detection (FastAPI + PyTorch)

A lightweight web API for **real-time object detection** using a trained Faster R-CNN model.  
Includes endpoints for image prediction, visualization, and live camera streaming—all optimized for smooth performance on both **Mac (M-series)** and **Windows**.

---

## 🧠 Features
- **REST API** built with FastAPI  
- **Image upload** detection endpoint (`/predict-image`)  
- **Visualization endpoint** returning annotated images (`/visualize-image`)  
- **Live camera streaming** (`/live`) with threaded inference  
- Compatible with **CPU, CUDA**, and **Apple Silicon (MPS)** backends  
- Docker-ready for quick deployment  

---

## 📦 Requirements

Python 3.10 +  

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

## ⚙️ Installation (Local – Mac/Windows)

```bash
# 1️⃣ Clone the repository
git clone https://github.com/yourusername/fasterrcnn-api.git
cd fasterrcnn-api

# 2️⃣ Create a virtual environment
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# OR
venv\Scripts\activate         # Windows

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

# Run container
docker run -p 8080:8080 fasterrcnn-api
```

To mount your trained model:
```bash
docker run -p 8080:8080 -v $(pwd)/models:/models fasterrcnn-api
```
> 🪟 On Windows PowerShell:
> ```bash
> docker run -p 8080:8080 -v ${PWD}/models:/models fasterrcnn-api
> ```

---

## 🔍 API Endpoints

### **1. Health Check**
**GET** `/health`  
Returns model device and server status.

### **2. Predict Image**
**POST** `/predict-image`  
Upload an image and receive JSON bounding-box predictions.  
```bash
curl -X POST "http://localhost:8080/predict-image"      -F "file=@sample.jpg"
```

### **3. Visualize Image**
**POST** `/visualize-image`  
Returns an annotated image (JPEG stream) with detected objects.  
```bash
curl -X POST "http://localhost:8080/visualize-image"      -F "file=@sample.jpg" --output output.jpg
```

### **4. Live Stream**
**GET** `/live`  
View real-time detection from your webcam.

---

## 🧩 Configuration

Edit `app/config.py` to set default parameters:

```python
DEVICE = "cuda" if torch.cuda.is_available()     else "mps" if torch.backends.mps.is_available() else "cpu"
NUM_THREADS = 4
MODEL_PATH = "models/best_model.pth"
SCORE_THRESH = 0.5
```

---

## ⚡ Performance Tips

- Use **MPS (Mac)** or **CUDA (Windows/Linux)** for faster inference.  
- Reduce frame size → lower latency (`T.Resize((320, 320))`).  
- Adjust frame skip in `generate_frames()` to control FPS vs speed.  
- Close extra browser tabs during live stream for smooth output.

---

## 🧠 Notes for macOS Users

If you see:
```
[ WARN:0] VIDEOIO(V4L2:/dev/video0): can't open camera by index
```
Replace camera index:
```python
cap = cv2.VideoCapture(1)
```
or grant camera permission:
> **System Settings → Privacy & Security → Camera → Allow Terminal / IDE access**

---

## 📁 Project Structure

```
.
├── app/
│   ├── main.py              # FastAPI application
│   ├── model.py             # Model loader
│   ├── predict_utils.py     # Preprocessing & filtering
│   ├── config.py            # App configuration
│   └── ...
├── models/
│   └── best_model.pth       # Trained weights
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧑‍💻 Author
**Glend Dale Ferrer**  
📧 mgdferrer@tip.edu.ph 


---

## 📜 License
MIT License © 2025 Glend Dale Ferrer
