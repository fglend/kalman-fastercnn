# -------------------------------
# 🐍 Base Image
# -------------------------------
FROM python:3.11-slim

# Disable buffering and ensure UTF-8
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

# -------------------------------
# 📂 Working directory
# -------------------------------
WORKDIR /app

# -------------------------------
# 📦 Install dependencies
# -------------------------------
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g localtunnel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# 📁 Copy application code
# -------------------------------
COPY . .

# -------------------------------
# 🧠 Environment variables
# -------------------------------
# Default directories for results and models
ENV RESULTS_DIR=/results
ENV MODEL_DIR=/models

# Ensure results directory exists (even without mount)
RUN mkdir -p $RESULTS_DIR $MODEL_DIR

# -------------------------------
# 🚪 Expose FastAPI port
# -------------------------------
EXPOSE 8080

# -------------------------------
# 🏃‍♂️ Run server and LocalTunnel
# -------------------------------
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8080 --timeout-keep-alive 600 & sleep 5 && lt --port 8080 --subdomain gd-live"]

# -------------------------------
# ❤️ Healthcheck (optional)
# -------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1