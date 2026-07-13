# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.11-slim

# The Track 2 harness does not inject API credentials. Bake the Fireworks key
# into the image so the container can authenticate without runtime env vars.
ARG FIREWORKS_API_KEY
ENV FIREWORKS_API_KEY=${FIREWORKS_API_KEY}

# Install FFmpeg, runtime libraries, and temporary build tools for faster-whisper/CTranslate2.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first for better layer caching.
COPY requirements-agent.txt .
RUN pip install --no-cache-dir -r requirements-agent.txt \
    && apt-get purge -y build-essential cmake \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Pre-download the whisper models so runtime is not spent fetching them.
ENV WHISPER_CACHE_DIR=/app/models
RUN python - <<'PY'
from faster_whisper import WhisperModel
for size in ("base", "tiny"):
    print(f"Pre-downloading whisper model: {size}")
    WhisperModel(size, device="cpu", compute_type="int8", download_root="/app/models")
PY

# Copy the application code.
COPY agent.py .
COPY config.py .
COPY schemas.py .
COPY pipeline/ ./pipeline/
COPY examples/ ./examples/

# Input/output mount points expected by the judging harness.
RUN mkdir -p /input /output

# Run the captioning agent on container start.
CMD ["python", "agent.py"]
