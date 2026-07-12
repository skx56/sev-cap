---
marp: true
theme: default
paginate: true
---

# SEV-Cap
## Grounded multi-style video captioning

Track 2 · four styles · accuracy + tone

`ghcr.io/skx56/sevcap-grounded:latest`

---

# Problem

Single-pass VLM captioners fail the judge on:

1. **Accuracy** — hallucinated details
2. **Tone** — style label without style content
3. **Harness** — wrong schema / missing tasks / timeouts

Combined score = `(mean_acc + mean_tone) / 10`

---

# Approach

1. One grounded description (describe → verify)
2. Multi-candidate captions per style
3. Vision prejudge on **accuracy + tone**
4. Polish / reselect weak styles
5. Anytime harness I/O

---

# Pipeline

`tasks.json` → keyframes → describe → verify → N drafts/style → prejudge → polish → `results.json`

---

# Styles

| Style | Voice |
| --- | --- |
| formal | Precise, neutral |
| sarcastic | Dry irony |
| humorous_tech | One tech metaphor |
| humorous_non_tech | Everyday humor, no jargon |

Same facts. Different voice.

---

# Harness

- In: `/input/tasks.json`
- Out: `/output/results.json`
- Image: `ghcr.io/skx56/sevcap-grounded:latest`
- linux/amd64 · argument-free · `FIREWORKS_API_KEY` at runtime

---

# Results (8 AMD sample clips)

**Combined 0.966** · mean acc 4.75 · mean tone 4.91

All 8 tasks `sev-verified` · schema OK · Docker OK

---

# Stack

Kimi K2.6 (Fireworks) · Python 3.11 · ffmpeg · Docker → GHCR

Repo: https://github.com/skx56/sev-cap
