# Use official Python image
FROM python:3.11-slim

# Prevent Python from buffering output
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Install Node.js and npm (required for Localtunnel)
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g localtunnel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirement file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose FastAPI port
EXPOSE 8080

# Run FastAPI and Localtunnel together
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8080 & sleep 5 && lt --port 8080 --subdomain gd-live"]