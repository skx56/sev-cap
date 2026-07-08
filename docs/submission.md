# Submission copy (paste into the hackathon form)

## Title

SEV-Cap: Semantic-Entropy Verified Video Captioning

## Short description

SEV-Cap captions videos in four styles (formal, sarcastic, humorous-tech,
humorous-non-tech) with a Gemma-3-only pipeline that pre-verifies itself on
the judge's two axes: facts survive Nature-2024 semantic-entropy verification
before captioning, and every caption must win a blind style lineup before it
ships.

## Long description

Single-pass video captioners fail the LLM-Judge on two predictable axes:
hallucinated details (accuracy) and style labels that are aspirational rather
than real (tone). SEV-Cap is architected against both.

Stage 1 samples 6-12 keyframes with ffmpeg (uniform + scene-change) and runs
K=5 independent fact extractions with Gemma 3 27B vision at temperature 0.7.
Following Farquhar et al. (Nature 2024), the atomic facts are clustered by
bidirectional entailment — with Gemma itself as the NLI judge — and per-fact
support across the K samples gives a semantic-entropy signal. Facts asserted
in fewer than 3 of 5 independent samples carry the confabulation signature the
Nature paper detects, and are rejected before captioning. To our knowledge
this is the first application of semantic-entropy verification to video
captioning: recent hallucination-mitigation work (GRAVITI, SEASON, SmartSight,
2025) grounds or contrasts decoding instead.

Stage 2 generates each of the four captions from the verified fact sheet
ONLY — the generator never sees the video, so unverified details cannot leak
in. Each caption set must then pass two gates: (A) grounding — every concrete
claim entailed by the fact sheet; (B) the blind style lineup — captions are
label-stripped, shuffled, and must be re-identified as their intended style
with confidence >= 3/5 by a fresh judge context. Failures are repaired via a
Self-Refine loop (Madaan et al., NeurIPS 2023) with the judge's verdict as
actionable feedback, capped at two rounds, best attempt kept.

The container is engineered for the scoring harness: linux/amd64, public
ghcr.io image, argument-free entrypoint, and an anytime algorithm — a valid
draft for every clip is written to disk immediately and upgraded in place with
atomic writes, so a timeout can degrade quality but never produce missing
output. Every clip's JSON includes a verification report: rejected
high-entropy facts, lineup verdicts and confidences, and retry history.

Gemma 3 27B is the extractor, entailment judge, caption writer, lineup judge
and refiner — one open model as engine and verifier, end-to-end.

## Tech stack tags

Gemma, Gemma 3 27B, Fireworks AI, Python, asyncio, ffmpeg, Docker

## Links

- Repo: https://github.com/<you>/sev-cap
- Container: ghcr.io/<you>/sev-cap:latest
- Cover image: assets/cover.png
- Slides: docs/slides.md (render with marp)

## Demo video script (~2 min)

1. (15s) Problem: single-pass captioners hallucinate and miss tone; the judge
   grades exactly those two axes.
2. (30s) Run `sevcap facts clips/bbb_action_45s.mp4` — show verified facts vs
   rejected high-entropy facts on screen ("semantic entropy caught these").
3. (30s) Show a caption failing the blind lineup, the Self-Refine feedback,
   and the improved rewrite passing.
4. (30s) `docker run` the exact submitted image end-to-end; show
   results/captions.json with the verification report.
5. (15s) Architecture slide + "one Gemma, engine and verifier."
