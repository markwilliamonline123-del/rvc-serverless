FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 1. Install system audio utilities cleanly
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy dependencies and upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Create persistent shared models directory
RUN mkdir -p /models

# 4. BULLETPROOF COPY: Grab all workspace files together 
# This prevents "file not found" errors even if GitHub sync is lagging.
COPY . .

# 5. Force listen on RunPod's assigned environmental port
EXPOSE 3000

# 6. Execute the unified app controller
CMD ["python", "-u", "app.py"]
