FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install critical system audio tools required by librosa and soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from the repository and install them
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create the permanent folder structure inside the container for the network volume
RUN mkdir -p /models

# Copy all the python script handlers into the working directory
COPY training_handler.py /app/training_handler.py
COPY inference_handler.py /app/inference_handler.py

# Default execution file safety fallback
CMD [ "python", "-u", "/app/inference_handler.py" ]
