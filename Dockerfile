# SEV-Cap scoring container. Build for the harness with:
#   docker buildx build --platform linux/amd64 -t ghcr.io/<you>/sevcap-grounded:latest .
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Defaults the harness can override with -e; the key is NEVER baked in.
# Audio ASR is off by default (vision-only scoring path). Multi-candidate
# caption selection + light polish maximize accuracy/tone on sample-like clips.
ENV INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    OUTPUT_PATH=/output/results.json \
    INPUT_PATH=/input/tasks.json \
    SEVCAP_CACHE=0 \
    SEVCAP_AUDIO=0 \
    SEVCAP_POLISH=1 \
    SEVCAP_POLISH_ROUNDS=1 \
    SEVCAP_CANDIDATES=3 \
    SEVCAP_CLIP_CONCURRENCY=2 \
    SEVCAP_LLM_CONCURRENCY=6 \
    SEVCAP_TIME_BUDGET=1200 \
    SEVCAP_MODEL="accounts/fireworks/models/kimi-k2p6" \
    SEVCAP_VISION_MODEL="accounts/fireworks/models/kimi-k2p6" \
    PYTHONUNBUFFERED=1

# Exec-form, argument-free: runs the full anytime pipeline.
ENTRYPOINT ["python", "-m", "sevcap"]
