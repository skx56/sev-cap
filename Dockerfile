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

# Defaults the harness can override with -e; the key is NEVER baked in.
# Primary model is our dedicated Gemma 4 31B deployment (scale-to-zero);
# if the scoring key cannot reach it, the client falls back to Kimi K2p6.
ENV INPUT_DIR=/input \
    OUTPUT_DIR=/output \
    SEVCAP_CACHE=0 \
    SEVCAP_MODEL="accounts/fireworks/models/gemma-4-26b-a4b-it#accounts/skx56/deployments/c4pafnfc" \
    SEVCAP_VISION_MODEL="accounts/fireworks/models/gemma-4-26b-a4b-it#accounts/skx56/deployments/c4pafnfc" \
    PYTHONUNBUFFERED=1

# Exec-form, argument-free: runs the full anytime pipeline.
ENTRYPOINT ["python", "-m", "sevcap"]
