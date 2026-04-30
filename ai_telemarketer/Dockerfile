FROM python:3.10-slim

WORKDIR /app

# System dependencies for TTS and Whisper
RUN apt-get update && apt-get install -y gcc g++ ffmpeg libsndfile1 git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Установка PyTorch с поддержкой CUDA (опционально, но желательно для GPU)
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY . .

EXPOSE 8000
CMD ["bash", "run.sh"]
