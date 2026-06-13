FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install critical system tools for audio data loading
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create space for permanent shared models volume mount
RUN mkdir -p /models

# Copy scripts over
COPY training_handler.py .
COPY inference_handler.py .
COPY app.py .

# Force unified app deployment entrypoint 
CMD ["python", "-u", "app.py"]
