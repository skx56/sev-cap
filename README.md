# SEV-Cap — multi-style video captioning (AMD Track 2)

Dockerized agent: reads `/input/tasks.json`, writes `/output/results.json`
with four captions per clip (`formal`, `sarcastic`, `humorous_tech`,
`humorous_non_tech`).

## Approach

1. Sample a small set of keyframes (3–6 by duration)
2. Vision model writes a structured scene brief, then verifies it
3. Text model writes the four styles sequentially with light style checks

## Docker image

`ghcr.io/skx56/sevcap-grounded:latest` (linux/amd64)

```bash
docker pull --platform linux/amd64 ghcr.io/skx56/sevcap-grounded:latest
docker run --rm \
  -v "$PWD/input:/input:ro" \
  -v "$PWD/output:/output" \
  ghcr.io/skx56/sevcap-grounded:latest
```

Defaults: MiniMax M3 (vision) + Kimi K2.6 (captions), ASR off, concurrency 3.

## Demo (Streamlit)

```bash
export FIREWORKS_API_KEY=fw_...
pip install -r requirements.txt
streamlit run demo/app.py
```

Or open the hosted Streamlit app linked to this repo (Community Cloud).
Set `FIREWORKS_API_KEY` in Streamlit secrets.

## Local harness

```bash
export FIREWORKS_API_KEY=fw_...
export INPUT_PATH=./sample_input/tasks.json
export OUTPUT_PATH=./out/results.json
python agent.py
```
