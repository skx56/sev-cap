# SEV-Cap — Grounded Multi-Style Video Captioning

Track 2 video captioning: four styles (**formal, sarcastic, humorous_tech,
humorous_non_tech**) optimized for the LLM judge’s **accuracy** and **tone**.

| | |
| --- | --- |
| **Docker image** | `ghcr.io/skx56/sevcap-grounded:latest` (linux/amd64) |
| **Repository** | <https://github.com/skx56/sev-cap> |
| **Presentation** | [SEV-Cap-Presentation.pdf](SEV-Cap-Presentation.pdf) |

## Scoring path (what the image runs)

1. Read `/input/tasks.json`, download/resolve each video
2. Sample keyframes (ffmpeg)
3. **Describe → verify** one shared scene description
4. Draft **multi-candidate** captions per style (+ vision-grounded drafts)
5. **Vision prejudge** on accuracy + tone; pick the best
6. Polish/reselect any weak styles
7. Write `/output/results.json` (`[{task_id, captions}, ...]`)

Tuned against the official AMD sample-style clips; internal judge combined
score **0.966** (`(mean_acc + mean_tone) / 10`).

## Quick start

```bash
git clone https://github.com/skx56/sev-cap && cd sev-cap
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
export FIREWORKS_API_KEY=fw_...
.venv/bin/sevcap check
```

```bash
docker pull --platform linux/amd64 ghcr.io/skx56/sevcap-grounded:latest
docker run --rm --platform linux/amd64 \
  -e FIREWORKS_API_KEY=fw_... \
  -v "$PWD/sample_input:/input:ro" -v "$PWD/results:/output" \
  ghcr.io/skx56/sevcap-grounded:latest
```

Submit: **`ghcr.io/skx56/sevcap-grounded:latest`**

## Repo layout

```
src/sevcap/     scoring pipeline
sample_input/   official-style tasks.json
eval/           internal judge
scripts/        harness schema validator
docs/           submission notes + slides
Dockerfile      linux/amd64 scoring image
```

## Config

See [.env.example](.env.example). Image defaults: `SEVCAP_CANDIDATES=3`,
`SEVCAP_POLISH=1`, `SEVCAP_AUDIO=0`, Kimi K2.6 on Fireworks.
