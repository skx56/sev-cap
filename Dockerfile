# SEV-Cap scoring container. Build for the harness with:
#   docker buildx build --platform linux/amd64 -t ghcr.io/<you>/sev-cap:latest .
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Bake the Whisper weights into the image at build time so the harness never
# needs extra network access at run time to transcribe audio (Fireworks
# deprecated its hosted Whisper endpoint, so this runs fully local/CPU).
ENV SEVCAP_WHISPER_CACHE_DIR=/app/.whisper_cache
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8', download_root='/app/.whisper_cache')"

# Defaults the harness can override with -e; the key is NEVER baked in.
# Primary is Kimi K2.6 (reliable serverless VLM+text). Gemma remains an
# optional env override for the bonus chase, not the image default.
ENV INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    OUTPUT_PATH=/output/results.json \
    INPUT_PATH=/input/tasks.json \
    SEVCAP_CACHE=0 \
    SEVCAP_AUDIO=0 \
    SEVCAP_POLISH=1 \
    SEVCAP_POLISH_ROUNDS=2 \
    SEVCAP_DOWNLOAD_CONCURRENCY=4 \
    SEVCAP_CLIP_CONCURRENCY=1 \
    SEVCAP_TIME_BUDGET=900 \
    SEVCAP_MODEL="accounts/fireworks/models/kimi-k2p6" \
    SEVCAP_VISION_MODEL="accounts/fireworks/models/kimi-k2p6" \
    PYTHONUNBUFFERED=1

# Exec-form, argument-free: runs the full anytime pipeline.
ENTRYPOINT ["python", "-m", "sevcap"]
