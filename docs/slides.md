---
marp: true
theme: default
class: invert
paginate: true
---

<!-- Render with: npx @marp-team/marp-cli docs/slides.md -o docs/slides.pdf -->

# SEV-Cap
## Semantic-Entropy Verified Video Captioning

Track 2 · Four styles, pre-verified on the two axes the judge grades

**Powered end-to-end by Gemma 3 27B** (Fireworks AI)

---

# The problem with single-pass captioning

Most pipelines: *VLM watches clip → writes 4 captions.*

The LLM-Judge catches them on exactly two axes:

- **Accuracy** — hallucinated objects, actions, on-screen text
- **Tone** — "sarcastic" that is just declarative-with-attitude,
  "humorous" that is a description with an exclamation mark

SEV-Cap attacks both **structurally**, not with prompt vibes.

---

# Twist 1 — Semantic-entropy verified facts

*Adapting Farquhar et al., "Detecting hallucinations in LLMs using semantic entropy", **Nature 2024** — to video captioning.*

1. Sample **K = 5 independent** fact extractions over keyframes (temp 0.7)
2. Cluster atomic facts by **bidirectional entailment** (Gemma as NLI judge)
3. Per-fact support across samples → semantic-entropy signal
4. Support ≥ 3/5 → **verified fact sheet**; support 1/5 → confabulation, **rejected & logged**

Captions are generated **from the fact sheet only** — the generator never
sees the video. An unverified detail *cannot* appear in a caption.

---

# Twist 2 — The blind style lineup

After generation, the 4 captions are **label-stripped and shuffled**.
A fresh judge context must match each caption to its style, with confidence 1-5.

- Misidentified or weak (< 3/5) → **fail**
- The judge's confusion ("this reads formal, not sarcastic") becomes
  actionable feedback for a **Self-Refine** rewrite (Madaan et al., NeurIPS 2023)
- Max 2 rounds, best attempt always kept

A caption ships only if it is **blindly identifiable as its intended style**.

---

# Architecture

```
clip → ffmpeg keyframes (uniform + scene-change)
     → instant 4-style draft ── written to disk immediately (anytime output)
     → K=5 fact extractions → entailment clustering → semantic entropy
     → verified fact sheet
     → 4 style generators (few-shot exemplars, facts-only context)
     → Gate A: grounding   Gate B: blind lineup
     → Self-Refine loop (≤2 rounds) → atomic overwrite of draft
```

**One model everywhere:** Gemma 3 27B is extractor, NLI judge, writer,
lineup judge and refiner — Gemma as its own verifier, not a bolt-on.

---

# Engineering for the scoring harness

40% of the field failed on logistics, not quality (PULL / RUNTIME / TIMEOUT / OUTPUT_MISSING).

- **linux/amd64** image, public ghcr.io, argument-free exec-form entrypoint
- **Anytime algorithm**: placeholder → draft → verified, each written atomically;
  a kill at any moment leaves valid output for every clip
- Global time budget, per-clip async concurrency, fail-fast on 4xx
- Verification report shipped per clip: rejected facts, lineup verdicts, retries

---

# Why this wins

- **Aimed at the grading**: pre-scores itself on accuracy (semantic entropy)
  and tone (blind lineup) before output leaves the container
- **Novel**: 2025 video-hallucination work (GRAVITI, SEASON, SmartSight)
  grounds decoding — none applies Nature-2024 semantic entropy to caption facts
- **Best use of Gemma**: one open model as engine *and* verifier, end-to-end
- **Receipts included**: every caption ships with its verification evidence
