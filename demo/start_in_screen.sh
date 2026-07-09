#!/bin/bash
cd /Users/saksham56/Developer/sev-cap
set -a
source .env
set +a
# Clear any Gemma override so the demo uses Kimi defaults from config.py
unset SEVCAP_MODEL SEVCAP_VISION_MODEL
exec .venv/bin/streamlit run demo/app.py --server.port "${SEVCAP_DEMO_PORT:-7860}" --server.headless true --server.address 0.0.0.0
