# Minimal CPU-only Dockerfile for third_eye runtime
# Uses Python 3.11 for broad wheel availability (Torch CPU)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    THIRD_EYE_PROFILE=balanced

# System deps for OpenCV and media I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure runtime directories exist
RUN mkdir -p logs recordings && \
    useradd -m -u 1000 app && \
    chown -R app:app /app

USER app

# Headless default; disable audio announcements in container
CMD ["bash", "scripts/launch.sh"]