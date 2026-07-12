# Submission copy (paste into the hackathon form)

## Title

SEV-Cap: Grounded Multi-Style Video Captioning

## Short description

SEV-Cap captions videos in four styles (formal, sarcastic, humorous-tech,
humorous-non-tech) with a grounded describe → verify → multi-candidate write
pipeline that pre-scores accuracy and tone before shipping captions.

## Long description

Single-pass video captioners fail the LLM-Judge on two axes: hallucinated
details (accuracy) and style labels that are aspirational (tone). SEV-Cap
attacks both structurally.

Keyframes are sampled with ffmpeg (uniform + scene-change). A shared scene
description is written and self-verified against the frames. Each style then
gets multiple caption candidates; a vision prejudge scores accuracy + tone and
keeps the best. Weak styles get a light polish / reselect pass when time
remains.

The container matches the Track 2 harness: linux/amd64, public ghcr.io image,
argument-free entrypoint, `/input/tasks.json` → `/output/results.json`, and an
anytime algorithm so timeouts degrade quality but never produce missing tasks.

Default model: Kimi K2.6 on Fireworks (override for Gemma bonus mode).

## Tech stack tags

Kimi K2.6, Fireworks AI, Python, asyncio, ffmpeg, Docker

## Links

- Repo: https://github.com/skx56/sev-cap
- Container: **ghcr.io/skx56/sevcap-grounded:latest**
- Slides: docs/slides.md
- Presentation: SEV-Cap-Presentation.pdf
