---
marp: true
theme: default
paginate: true
---

# SEV-Cap
## Grounded multi-style video captioning

Track 2 · four styles · accuracy + tone

---

# Problem

Single-pass VLM captioners fail the judge on:

1. **Accuracy** — hallucinated details
2. **Tone** — style label without style content

---

# Pipeline

1. Sample keyframes (ffmpeg)
2. Describe → self-verify scene
3. Multi-candidate captions per style
4. Vision prejudge (accuracy + tone) → pick best
5. Optional polish / reselect

---

# Harness contract

- Input: `/input/tasks.json`
- Output: `/output/results.json`
- Image: `ghcr.io/skx56/sevcap-grounded:latest`
- linux/amd64 · argument-free entrypoint · anytime writes

---

# Styles

formal · sarcastic · humorous_tech · humorous_non_tech

---

# Stack

Kimi K2.6 (Fireworks) · Python 3.11 · ffmpeg · Docker
