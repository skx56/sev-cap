"""Generate the SEV-Cap project presentation as an 8-page PDF.

Run:  .venv/bin/python scripts/make_deck.py
Out:  SEV-Cap-Presentation.pdf  (repo root)
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "SEV-Cap-Presentation.pdf"

W, H = landscape(A4)  # 842 x 595 pt

# palette
BG = HexColor("#0b0b14")
CARD = HexColor("#15151f")
CARD2 = HexColor("#1c1c2b")
INK = HexColor("#e7e7f0")
MUTE = HexColor("#9a9ab0")
VIOLET = HexColor("#a78bfa")
PINK = HexColor("#ec4899")
CYAN = HexColor("#22d3ee")
AMBER = HexColor("#f59e0b")
GREEN = HexColor("#34d399")
RED = HexColor("#f87171")

MARGIN = 46


def bg(c: canvas.Canvas) -> None:
    c.setFillColor(BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    # subtle top accent bar
    c.setFillColor(VIOLET)
    c.rect(0, H - 6, W, 6, fill=1, stroke=0)


def footer(c: canvas.Canvas, n: int) -> None:
    c.setFillColor(MUTE)
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN, 24, "SEV-Cap  ·  Semantic-Entropy Verified Video Captioning")
    c.drawRightString(W - MARGIN, 24, f"{n} / 8")


def kicker(c: canvas.Canvas, text: str, color=VIOLET) -> None:
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN, H - 66, text.upper())


def title(c: canvas.Canvas, text: str, y=H - 104) -> None:
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 30)
    c.drawString(MARGIN, y, text)


def wrap(c: canvas.Canvas, text, x, y, width, font="Helvetica", size=12,
         leading=17, color=INK):
    c.setFillColor(color)
    c.setFont(font, size)
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, font, size) <= width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= leading
            line = w
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


def card(c: canvas.Canvas, x, y, w, h, fill=CARD, radius=12, accent=None):
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, radius, fill=1, stroke=0)
    if accent:
        c.setFillColor(accent)
        c.roundRect(x, y + h - 5, w, 5, 2.5, fill=1, stroke=0)


def bullet(c: canvas.Canvas, x, y, head, body, accent=VIOLET, width=None):
    c.setFillColor(accent)
    c.circle(x + 4, y + 4, 4, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x + 16, y, head)
    if body:
        yy = y - 17
        return wrap(c, body, x + 16, yy, width or (W - MARGIN - x - 16),
                    size=11.5, leading=15.5, color=MUTE)
    return y - 18


# ---------------------------------------------------------------- slides
def slide_title(c):
    bg(c)
    c.setFillColor(VIOLET)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(MARGIN, H - 150, "GEMMA HACKATHON  ·  TRACK 2  ·  VIDEO CAPTIONING")

    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 68)
    c.drawString(MARGIN, H - 235, "SEV-Cap")

    c.setFillColor(CYAN)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(MARGIN, H - 275, "Semantic-Entropy Verified Video Captioning")

    wrap(c,
         "Four caption styles per clip — formal, sarcastic, humorous-tech, "
         "humorous-non-tech — each pre-verified on the same two axes the judge "
         "grades: factual accuracy and stylistic tone.",
         MARGIN, H - 315, W * 0.62, size=13, leading=19, color=MUTE)

    # style chips
    chips = [("Formal", VIOLET), ("Sarcastic", PINK),
             ("Humorous · Tech", CYAN), ("Humorous · Non-Tech", AMBER)]
    x = MARGIN
    y = 120
    for label, col in chips:
        w = c.stringWidth(label, "Helvetica-Bold", 12) + 30
        c.setFillColor(CARD2)
        c.roundRect(x, y, w, 30, 15, fill=1, stroke=0)
        c.setFillColor(col)
        c.circle(x + 15, y + 15, 4, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x + 26, y + 10, label)
        x += w + 12

    c.setFillColor(MUTE)
    c.setFont("Helvetica", 11)
    c.drawString(MARGIN, 78, "Live demo:  associates-desired-avatar-doctrine.trycloudflare.com")
    c.drawString(MARGIN, 60, "Code:  github.com/skx56/sev-cap")
    footer(c, 1)
    c.showPage()


def slide_problem(c):
    bg(c)
    kicker(c, "The problem", PINK)
    title(c, "Where single-pass captioners lose points")
    wrap(c,
         "An LLM-Judge grades every caption on two axes. A naive 'VLM watches "
         "the clip and writes four captions' pipeline fails on both, predictably.",
         MARGIN, H - 138, W - 2 * MARGIN, size=13, leading=18, color=MUTE)

    cw = (W - 2 * MARGIN - 24) / 2
    cx1, cx2 = MARGIN, MARGIN + cw + 24
    cy, ch = 150, 250
    card(c, cx1, cy, cw, ch, accent=RED)
    card(c, cx2, cy, cw, ch, accent=AMBER)

    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(cx1 + 22, cy + ch - 42, "1.  Accuracy — hallucination")
    wrap(c,
         "The model invents plausible details that are not in the video: an "
         "object that is not there, a colour that is wrong, an action that never "
         "happened. These read fluently and are hard to catch after the fact.",
         cx1 + 22, cy + ch - 72, cw - 44, size=12, leading=17, color=MUTE)

    c.setFillColor(AMBER)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(cx2 + 22, cy + ch - 42, "2.  Tone — aspirational style")
    wrap(c,
         "A caption labelled 'sarcastic' that actually reads as flat and "
         "declarative. The style tag is a label the model hoped for, not a "
         "property the text demonstrably has.",
         cx2 + 22, cy + ch - 72, cw - 44, size=12, leading=17, color=MUTE)

    wrap(c,
         "SEV-Cap's thesis: don't hope — verify. Attack each failure mode with a "
         "dedicated check before the caption is ever emitted.",
         MARGIN, 118, W - 2 * MARGIN, font="Helvetica-Bold", size=14,
         leading=19, color=INK)
    footer(c, 2)
    c.showPage()


def slide_approach(c):
    bg(c)
    kicker(c, "The idea", CYAN)
    title(c, "Verify first, caption second")
    wrap(c,
         "A two-stage pipeline with a hallucination firewall between them. Facts "
         "are established and verified before any caption is written; captions "
         "are then generated and must pass two gates.",
         MARGIN, H - 138, W - 2 * MARGIN, size=13, leading=18, color=MUTE)

    cw = (W - 2 * MARGIN - 24) / 2
    cx1, cx2 = MARGIN, MARGIN + cw + 24
    cy, ch = 150, 250
    card(c, cx1, cy, cw, ch, accent=VIOLET)
    card(c, cx2, cy, cw, ch, accent=GREEN)

    c.setFillColor(VIOLET)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(cx1 + 22, cy + ch - 40, "Stage 1 — Establish facts")
    yy = cy + ch - 70
    for t in [
        "Sample keyframes (ffmpeg) + local audio transcript (Whisper).",
        "K=3 independent extractions, not one trusted pass.",
        "Semantic-entropy verification filters confabulations.",
        "Output: a verified fact sheet with per-fact support.",
    ]:
        c.setFillColor(VIOLET); c.circle(cx1 + 26, yy + 4, 3, fill=1, stroke=0)
        yy = wrap(c, t, cx1 + 36, yy, cw - 58, size=11.5, leading=15,
                  color=MUTE) - 4

    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(cx2 + 22, cy + ch - 40, "Stage 2 — Write & prove")
    yy = cy + ch - 70
    for t in [
        "Generate 4 styles from the fact sheet + keyframes.",
        "Gate A: every claim must be grounded in the fact sheet.",
        "Gate B: blind style lineup must re-identify each style.",
        "Failures repaired via a Self-Refine loop with feedback.",
    ]:
        c.setFillColor(GREEN); c.circle(cx2 + 26, yy + 4, 3, fill=1, stroke=0)
        yy = wrap(c, t, cx2 + 36, yy, cw - 58, size=11.5, leading=15,
                  color=MUTE) - 4

    # firewall label between
    c.setFillColor(CARD2)
    c.roundRect(W / 2 - 70, cy + ch / 2 - 16, 140, 32, 16, fill=1, stroke=0)
    c.setFillColor(CYAN)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(W / 2, cy + ch / 2 - 4, "FACT FIREWALL")

    footer(c, 3)
    c.showPage()


def slide_stage1(c):
    bg(c)
    kicker(c, "Stage 1 · accuracy", VIOLET)
    title(c, "Semantic-entropy fact verification")
    wrap(c,
         "Adapted from Farquhar et al., Nature 2024 — hallucinations show up as "
         "high semantic entropy across independent samples. We apply that signal "
         "to a captioning fact set.",
         MARGIN, H - 138, W - 2 * MARGIN, size=12.5, leading=17, color=MUTE)

    x = MARGIN
    y = 300
    steps = [
        ("K=3 sampling", "Extract atomic facts 3× independently over keyframes + transcript.", VIOLET),
        ("Entailment clustering", "Group facts by bidirectional entailment (A⇔B) via an LLM NLI judge.", CYAN),
        ("Entropy threshold", "Keep facts with support ≥ 2 of 3. Singletons = confabulation → rejected.", GREEN),
        ("Verified fact sheet", "Only surviving facts move forward; rejects are logged for the report.", AMBER),
    ]
    bw = (W - 2 * MARGIN - 3 * 16) / 4
    for i, (head, body, col) in enumerate(steps):
        cx = x + i * (bw + 16)
        card(c, cx, y, bw, 150, fill=CARD, accent=col)
        c.setFillColor(col)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(cx + 14, y + 118, head)
        wrap(c, body, cx + 14, y + 96, bw - 28, size=10.5, leading=14, color=MUTE)
        if i < 3:
            c.setFillColor(MUTE)
            c.setFont("Helvetica-Bold", 20)
            c.drawCentredString(cx + bw + 8, y + 70, "›")

    card(c, MARGIN, 120, W - 2 * MARGIN, 150, fill=CARD2)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(MARGIN + 20, 244, "Two evidence sources, not one")
    wrap(c,
         "Vision keyframes catch what is shown; a local faster-whisper transcript "
         "(CPU, int8, weights baked into the Docker image) catches dialogue and "
         "sound cues the frames miss. Both feed the same K-sample extraction, so "
         "audio-grounded facts get the same entropy verification as visual ones.",
         MARGIN + 20, 218, W - 2 * MARGIN - 40, size=12, leading=17, color=MUTE)
    footer(c, 4)
    c.showPage()


def slide_stage2(c):
    bg(c)
    kicker(c, "Stage 2 · tone", GREEN)
    title(c, "Two gates + Self-Refine")
    wrap(c,
         "Captions are generated from the verified fact sheet and keyframes, then "
         "must prove both their accuracy and their style before being kept.",
         MARGIN, H - 138, W - 2 * MARGIN, size=12.5, leading=17, color=MUTE)

    cw = (W - 2 * MARGIN - 24) / 2
    cx1, cx2 = MARGIN, MARGIN + cw + 24
    cy, ch = 210, 190
    card(c, cx1, cy, cw, ch, accent=CYAN)
    card(c, cx2, cy, cw, ch, accent=PINK)

    c.setFillColor(CYAN)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(cx1 + 22, cy + ch - 40, "Gate A — Grounding")
    wrap(c,
         "Every concrete claim in a caption must be entailed by the fact sheet. "
         "An unsupported claim means the caption is regenerated with the "
         "offending phrase handed back as feedback.",
         cx1 + 22, cy + ch - 68, cw - 44, size=12, leading=16.5, color=MUTE)

    c.setFillColor(PINK)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(cx2 + 22, cy + ch - 40, "Gate B — Blind style lineup")
    wrap(c,
         "The four captions are label-stripped, shuffled, and shown to a fresh "
         "judge that must re-identify each style with confidence ≥ 3/5. A style "
         "that doesn't read as itself fails.",
         cx2 + 22, cy + ch - 68, cw - 44, size=12, leading=16.5, color=MUTE)

    card(c, MARGIN, 110, W - 2 * MARGIN, 78, fill=CARD2, accent=AMBER)
    c.setFillColor(AMBER)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN + 20, 158, "Self-Refine loop (Madaan et al., NeurIPS 2023)")
    wrap(c,
         "Any failed caption is rewritten using the judge's specific feedback, "
         "capped at one round — most of the grounding/style win at roughly half "
         "the Stage-2 cost. If a caption still can't pass, the reliable draft is kept.",
         MARGIN + 20, 134, W - 2 * MARGIN - 40, size=11.5, leading=15.5, color=MUTE)
    footer(c, 5)
    c.showPage()


def slide_arch(c):
    bg(c)
    kicker(c, "Architecture", CYAN)
    title(c, "End-to-end pipeline")

    def box(x, y, w, h, label, col, sub=None, fill=CARD):
        card(c, x, y, w, h, fill=fill, accent=col, radius=10)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 11.5)
        c.drawCentredString(x + w / 2, y + h - 20, label)
        if sub:
            c.setFillColor(MUTE)
            c.setFont("Helvetica", 9)
            c.drawCentredString(x + w / 2, y + h - 34, sub)

    def arrow(x1, y1, x2, y2):
        c.setStrokeColor(MUTE)
        c.setLineWidth(1.4)
        c.line(x1, y1, x2, y2)
        # arrowhead
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        for da in (2.6, -2.6):
            c.line(x2, y2, x2 - 7 * math.cos(ang - da / 3), y2 - 7 * math.sin(ang - da / 3))

    top = H - 150
    # input column
    box(MARGIN, top - 40, 150, 40, "Video clip", VIOLET, "30s – 2 min")
    box(MARGIN, top - 120, 150, 46, "ffmpeg keyframes", CYAN, "uniform + scene-cut")
    box(MARGIN, top - 190, 150, 46, "Whisper transcript", CYAN, "local CPU ASR")
    arrow(MARGIN + 75, top - 40, MARGIN + 75, top - 74)
    arrow(MARGIN + 75, top - 40, MARGIN + 75, top - 144)

    # draft (anytime)
    box(MARGIN + 190, top - 120, 150, 46, "Instant draft", AMBER, "4 styles, atomic write")
    arrow(MARGIN + 150, top - 97, MARGIN + 190, top - 97)

    # stage 1
    box(MARGIN + 190, top - 200, 150, 46, "K=3 extraction", VIOLET, "vision + audio")
    box(MARGIN + 380, top - 200, 150, 46, "Entailment cluster", VIOLET, "LLM NLI judge")
    box(MARGIN + 570, top - 200, 150, 46, "Semantic entropy", GREEN, "keep ≥ 2 of 3")
    arrow(MARGIN + 150, top - 167, MARGIN + 190, top - 177)
    arrow(MARGIN + 340, top - 177, MARGIN + 380, top - 177)
    arrow(MARGIN + 530, top - 177, MARGIN + 570, top - 177)

    # verified sheet
    box(MARGIN + 570, top - 120, 150, 46, "Verified fact sheet", GREEN)
    arrow(MARGIN + 645, top - 200, MARGIN + 645, top - 154)

    # stage 2
    box(MARGIN + 570, top - 300, 150, 46, "4-style generator", CYAN, "grounded on facts+frames")
    arrow(MARGIN + 645, top - 154, MARGIN + 720, top - 154)
    arrow(MARGIN + 720, top - 154, MARGIN + 720, top - 277)
    arrow(MARGIN + 720, top - 277, MARGIN + 645, top - 277)

    box(MARGIN + 380, top - 300, 150, 46, "Gate A + Gate B", PINK, "grounding + lineup")
    box(MARGIN + 190, top - 300, 150, 46, "Self-Refine ×1", AMBER, "judge feedback")
    arrow(MARGIN + 570, top - 277, MARGIN + 530, top - 277)
    arrow(MARGIN + 380, top - 277, MARGIN + 340, top - 277)

    box(MARGIN, top - 300, 150, 46, "Final output", GREEN, "captions + report")
    arrow(MARGIN + 190, top - 277, MARGIN + 150, top - 277)

    card(c, MARGIN, 74, W - 2 * MARGIN, 40, fill=CARD2)
    c.setFillColor(MUTE)
    c.setFont("Helvetica", 11)
    c.drawString(MARGIN + 18, 90,
                 "Anytime guarantee: the draft is written immediately and only "
                 "overwritten if the upgraded captions pass both gates within the "
                 "per-clip timeout — output is never missing.")
    footer(c, 6)
    c.showPage()


def slide_eng(c):
    bg(c)
    kicker(c, "Engineering", AMBER)
    title(c, "Built to survive an automated judge")

    items = [
        ("Anytime algorithm", "Draft written the moment frames are sampled; "
         "verified pipeline upgrades in place with atomic writes. Timeouts "
         "degrade quality, never produce OUTPUT_MISSING.", AMBER),
        ("Hard per-clip timeout", "A stuck clip abandons its upgrade and keeps "
         "the draft rather than starving the global time budget.", RED),
        ("Model fallback", "Kimi K2.6 by default; if a call degenerates or a "
         "Gemma deployment is cold, it fails over automatically — no dead run.", CYAN),
        ("Container-first", "linux/amd64 image, argument-free entrypoint, "
         "auto-published to ghcr.io via GitHub Actions on every push.", VIOLET),
        ("Baked-in ASR", "Whisper weights baked at build time — audio "
         "transcription needs zero extra network at run time.", GREEN),
        ("Shows its receipts", "Every clip JSON carries a verification report: "
         "verified facts, rejected high-entropy facts, lineup verdicts, retries.", PINK),
    ]
    cw = (W - 2 * MARGIN - 2 * 20) / 3
    ch = 150
    for i, (head, body, col) in enumerate(items):
        row, coln = divmod(i, 3)
        cx = MARGIN + coln * (cw + 20)
        cy = 300 - row * (ch + 20)
        card(c, cx, cy, cw, ch, fill=CARD, accent=col)
        c.setFillColor(col)
        c.setFont("Helvetica-Bold", 13.5)
        c.drawString(cx + 16, cy + ch - 30, head)
        wrap(c, body, cx + 16, cy + ch - 54, cw - 32, size=11, leading=15, color=MUTE)
    footer(c, 7)
    c.showPage()


def slide_results(c):
    bg(c)
    kicker(c, "Results & links", GREEN)
    title(c, "Verified captions, measured")

    stats = [("0.86", "Combined score", VIOLET),
             ("3.82 / 5", "Mean accuracy", CYAN),
             ("4.79 / 5", "Mean tone", GREEN),
             ("7", "Eval clips", AMBER)]
    bw = (W - 2 * MARGIN - 3 * 18) / 4
    for i, (v, k, col) in enumerate(stats):
        cx = MARGIN + i * (bw + 18)
        card(c, cx, 320, bw, 110, fill=CARD, accent=col)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 30)
        c.drawString(cx + 16, 372, v)
        c.setFillColor(MUTE)
        c.setFont("Helvetica", 11)
        c.drawString(cx + 16, 348, k)

    wrap(c,
         "Scored by an LLM-Judge on Big Buck Bunny + Elephants Dream (CC-BY), on "
         "the same accuracy/tone axes the pipeline verifies against internally.",
         MARGIN, 292, W - 2 * MARGIN, size=12, leading=16, color=MUTE)

    card(c, MARGIN, 150, W - 2 * MARGIN, 120, fill=CARD2)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(MARGIN + 22, 234, "One open model, engine and verifier")
    wrap(c,
         "Kimi K2.6 runs everything by default; the whole pipeline re-points to "
         "Gemma 4 26B (on-demand, scale-to-zero) with a single env var to chase "
         "the Best-Use-of-Gemma bonus, with Kimi as an instant safety net. A "
         "Streamlit demo lets anyone upload a clip and inspect the verification "
         "report live.",
         MARGIN + 22, 208, W - 2 * MARGIN - 44, size=12, leading=16.5, color=MUTE)

    c.setFillColor(VIOLET)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN, 112, "Live demo:")
    c.setFillColor(INK); c.setFont("Helvetica", 12)
    c.drawString(MARGIN + 78, 112, "associates-desired-avatar-doctrine.trycloudflare.com")
    c.setFillColor(VIOLET); c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN, 92, "Code:")
    c.setFillColor(INK); c.setFont("Helvetica", 12)
    c.drawString(MARGIN + 78, 92, "github.com/skx56/sev-cap")

    c.setFillColor(MUTE); c.setFont("Helvetica-Oblique", 9)
    c.drawString(MARGIN, 64,
                 "Refs: Farquhar et al. (Nature 2024) · Madaan et al. (NeurIPS 2023) · "
                 "Kuhn et al. (ICLR 2023)")
    footer(c, 8)
    c.showPage()


def main():
    c = canvas.Canvas(str(OUT), pagesize=landscape(A4))
    c.setTitle("SEV-Cap — Project Presentation")
    slide_title(c)
    slide_problem(c)
    slide_approach(c)
    slide_stage1(c)
    slide_stage2(c)
    slide_arch(c)
    slide_eng(c)
    slide_results(c)
    c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
