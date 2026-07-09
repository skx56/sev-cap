# 90-second demo recording script

Record with **Cmd+Shift+5** (macOS) → "Record Selected Portion" → select your
terminal window. Stop with the menu-bar stop button. Upload the resulting
`.mov` to YouTube (Unlisted is fine — judges just need the link) or Loom,
then paste that link into the "Demo Application URL" field
(Platform = YouTube or Loom, whichever you use).

Total run time below is ~70s of actual pipeline work; talk over it live or
record narration after. Run everything from the repo root with your
`FIREWORKS_API_KEY` exported.

## Setup (do this once, before hitting record)

```bash
cd ~/Developer/sev-cap
export FIREWORKS_API_KEY=fw_...       # your key
rm -rf .sevcap_cache /tmp/demo_out
mkdir -p /tmp/demo_in
cp clips/ed_dialogue_60s.mp4 /tmp/demo_in/
```

## Script (say this while the commands run)

**[0:00–0:10] Hook**
> "Most video captioners are single-pass — a VLM watches a clip and writes
> four captions, and it hallucinates details and fakes the tone. SEV-Cap
> pre-verifies itself on both axes before anything ships."

**[0:10–0:20] Show the fact verification directly**

```bash
.venv/bin/sevcap facts clips/ed_dialogue_60s.mp4
```

> "This clip has dialogue — Whisper transcribes it locally, feeds it to five
> independent Gemma vision extractions alongside the frames. Only facts that
> multiple independent samples agree on survive — that's semantic-entropy
> verification, adapted from a Nature 2024 hallucination-detection paper.
> You can see the audio transcript logged, and the verified vs. rejected
> facts below it."

**[0:20–0:55] Run the real container end-to-end**

```bash
docker run --rm --platform linux/amd64 \
  -e FIREWORKS_API_KEY \
  -v /tmp/demo_in:/input:ro -v /tmp/demo_out:/output \
  ghcr.io/skx56/sev-cap:latest
```

> "This is the exact public image the judges pull — same
> `ghcr.io/skx56/sev-cap:latest` tag, linux/amd64, no arguments. It writes a
> draft caption to disk immediately for every clip — so a harness timeout can
> never leave output missing — then upgrades it in place once the verified
> fact sheet, style gates, and self-refine loop all pass."

**[0:55–1:10] Show the output + verification report**

```bash
cat /tmp/demo_out/ed_dialogue_60s.json | python3 -m json.tool | head -40
```

> "Four styles — formal, sarcastic, humorous-tech, humorous-non-tech — each
> one only as good as the fact sheet underneath it. And every clip ships
> with a verification report: which facts were rejected as high-entropy,
> which style gates passed, how many refine rounds it took. The pipeline
> shows its receipts."

**[1:10–1:15] Close**

> "One Gemma deployment — extraction, entailment judging, generation, style
> lineup, and refinement — end to end, with automatic fallback if it's ever
> unreachable. That's SEV-Cap."

## After recording

1. Stop the recording, it saves to your Desktop as `Screen Recording ....mov`.
2. Upload to YouTube (Settings → Unlisted) or Loom.
3. Copy the share link into the hackathon form:
   - **Demo Application Platform:** YouTube (or Loom)
   - **Demo Application URL:** the link
