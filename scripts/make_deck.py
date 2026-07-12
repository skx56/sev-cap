"""Generate the SEV-Cap project presentation PDF (8 landscape pages).

Run:  .venv/bin/python scripts/make_deck.py
Out:  SEV-Cap-Presentation.pdf
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import landscape, A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "SEV-Cap-Presentation.pdf"
W, H = landscape(A4)  # 842 x 595

# Clean light deck — slate ink, teal accent (not purple/cream AI defaults)
BG = HexColor("#F7F8FA")
CARD = HexColor("#FFFFFF")
INK = HexColor("#0F172A")
MUTE = HexColor("#475569")
TEAL = HexColor("#0D9488")
TEAL_DK = HexColor("#0F766E")
AMBER = HexColor("#D97706")
RED = HexColor("#DC2626")
LINE = HexColor("#E2E8F0")

MARGIN = 48


def bg(c: canvas.Canvas) -> None:
    c.setFillColor(BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.rect(0, H - 8, W, 8, fill=1, stroke=0)


def footer(c: canvas.Canvas, n: int, total: int = 8) -> None:
    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.line(MARGIN, 40, W - MARGIN, 40)
    c.setFillColor(MUTE)
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN, 22, "SEV-Cap  ·  Grounded Multi-Style Video Captioning")
    c.drawRightString(W - MARGIN, 22, f"{n} / {total}")


def kicker(c: canvas.Canvas, text: str) -> None:
    c.setFillColor(TEAL_DK)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, H - 56, text.upper())


def title(c: canvas.Canvas, text: str, y: float | None = None) -> None:
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 28)
    c.drawString(MARGIN, y if y is not None else H - 96, text)


def wrap(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    font: str = "Helvetica",
    size: float = 12,
    leading: float = 17,
    color=INK,
) -> float:
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


def card(c: canvas.Canvas, x: float, y: float, w: float, h: float, accent=None) -> None:
    c.setFillColor(CARD)
    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.roundRect(x, y, w, h, 10, fill=1, stroke=1)
    if accent:
        c.setFillColor(accent)
        c.roundRect(x, y + h - 4, w, 4, 2, fill=1, stroke=0)


def pill(c: canvas.Canvas, x: float, y: float, text: str, fill=TEAL) -> float:
    c.setFont("Helvetica-Bold", 10)
    tw = c.stringWidth(text, "Helvetica-Bold", 10) + 18
    c.setFillColor(fill)
    c.roundRect(x, y, tw, 22, 11, fill=1, stroke=0)
    c.setFillColor(HexColor("#FFFFFF"))
    c.drawString(x + 9, y + 7, text)
    return tw


# ---------------------------------------------------------------- slides
def slide_title(c: canvas.Canvas) -> None:
    bg(c)
    c.setFillColor(TEAL_DK)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN, H - 120, "AMD / GEMMA HACKATHON  ·  TRACK 2  ·  VIDEO CAPTIONING")

    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 64)
    c.drawString(MARGIN, H - 210, "SEV-Cap")

    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(MARGIN, H - 250, "Grounded Multi-Style Video Captioning")

    wrap(
        c,
        "Four captions per clip — formal, sarcastic, humorous-tech, "
        "humorous-non-tech — optimized for the same two axes the LLM judge "
        "grades: factual accuracy and stylistic tone.",
        MARGIN, H - 300, W - 2 * MARGIN, size=14, leading=20, color=MUTE,
    )

    x = MARGIN
    for label in ("accuracy", "tone", "anytime I/O", "linux/amd64"):
        x += pill(c, x, 90, label) + 10
    footer(c, 1)


def slide_problem(c: canvas.Canvas) -> None:
    bg(c)
    kicker(c, "The problem")
    title(c, "Single-pass captioners fail the judge")

    wrap(
        c,
        "Track 2 scores each caption on accuracy (1–5) and tone (1–5). "
        "Combined leaderboard score is (mean accuracy + mean tone) / 10.",
        MARGIN, H - 130, W - 2 * MARGIN, size=13, leading=18, color=MUTE,
    )

    cards = [
        (RED, "Accuracy fails",
         "Models invent cities, brands, camera moves, and plot that never appear in the frames."),
        (AMBER, "Tone fails",
         "A “sarcastic” caption that is just a formal paraphrase, or humor with no joke."),
        (TEAL, "Harness fails",
         "Wrong schema, missing tasks, or timeouts → OUTPUT_MISSING / INVALID_RESULTS."),
    ]
    cw = (W - 2 * MARGIN - 24) / 3
    for i, (accent, head, body) in enumerate(cards):
        x = MARGIN + i * (cw + 12)
        card(c, x, 90, cw, 250, accent=accent)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(x + 16, 300, head)
        wrap(c, body, x + 16, 270, cw - 32, size=12, leading=16, color=MUTE)
    footer(c, 2)


def slide_approach(c: canvas.Canvas) -> None:
    bg(c)
    kicker(c, "Approach")
    title(c, "Optimize for how the judge actually scores")

    rows = [
        ("01", "One grounded description",
         "Describe the scene from keyframes, then self-verify against the frames before any styled writing."),
        ("02", "Multi-candidate styles",
         "Draft several captions per style (text + vision-grounded). Never ship the first draft blindly."),
        ("03", "Vision prejudge",
         "Score each candidate on accuracy + tone — the same axes as Track 2 — and keep the best."),
        ("04", "Polish weak styles",
         "If a style is still soft, reselect with safer / vision-grounded rewrites before time runs out."),
    ]
    y = H - 145
    for num, head, body in rows:
        c.setFillColor(TEAL)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(MARGIN, y, num)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(MARGIN + 48, y, head)
        wrap(c, body, MARGIN + 48, y - 22, W - 2 * MARGIN - 48, size=12, leading=16, color=MUTE)
        y -= 88
    footer(c, 3)


def slide_pipeline(c: canvas.Canvas) -> None:
    bg(c)
    kicker(c, "Architecture")
    title(c, "End-to-end scoring pipeline")

    steps = [
        ("tasks.json", "Discover & download"),
        ("ffmpeg", "Keyframes"),
        ("describe", "Scene draft"),
        ("verify", "Fact check"),
        ("N drafts", "Per style"),
        ("prejudge", "acc + tone"),
        ("polish", "Reselect"),
        ("results.json", "Harness out"),
    ]
    n = len(steps)
    gap = 8
    box_w = (W - 2 * MARGIN - gap * (n - 1)) / n
    y0 = H - 280
    for i, (top, bot) in enumerate(steps):
        x = MARGIN + i * (box_w + gap)
        card(c, x, y0, box_w, 110, accent=TEAL if i in (5, 7) else None)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 9)
        # wrap short labels
        c.drawCentredString(x + box_w / 2, y0 + 68, top)
        c.setFillColor(MUTE)
        c.setFont("Helvetica", 8)
        c.drawCentredString(x + box_w / 2, y0 + 48, bot)
        if i < n - 1:
            c.setFillColor(TEAL)
            c.setFont("Helvetica-Bold", 14)
            c.drawString(x + box_w + 1, y0 + 50, "›")

    bullets = [
        "Shared description anchors every style so jokes stay on the same facts.",
        "Prejudge failures retry with JSON/regex parsing — no fake mid-scores that hide bad picks.",
        "Anytime writes: placeholders early, atomic upgrades, finalize every task_id.",
    ]
    y = 200
    for b in bullets:
        c.setFillColor(TEAL)
        c.circle(MARGIN + 5, y + 4, 4, fill=1, stroke=0)
        wrap(c, b, MARGIN + 18, y, W - 2 * MARGIN - 18, size=12, leading=16, color=INK)
        y -= 40
    footer(c, 4)


def slide_styles(c: canvas.Canvas) -> None:
    bg(c)
    kicker(c, "Outputs")
    title(c, "Four styles — same facts, different voice")

    styles = [
        (TEAL, "Formal", "Precise, neutral, news-desk prose. Concrete visible details only."),
        (AMBER, "Sarcastic", "Dry ironic praise or understatement. Tone without invented plot."),
        (HexColor("#2563EB"), "Humorous · Tech", "One tech metaphor mapped onto the primary on-screen action."),
        (HexColor("#7C3AED"), "Humorous · Non-Tech", "Warm everyday personification. Zero jargon, no invented drama."),
    ]
    cw = (W - 2 * MARGIN - 16) / 2
    ch = 150
    for i, (accent, name, body) in enumerate(styles):
        x = MARGIN + (i % 2) * (cw + 16)
        y = H - 300 if i < 2 else 90
        card(c, x, y, cw, ch, accent=accent)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(x + 18, y + ch - 40, name)
        wrap(c, body, x + 18, y + ch - 70, cw - 36, size=12, leading=16, color=MUTE)
    footer(c, 5)


def slide_harness(c: canvas.Canvas) -> None:
    bg(c)
    kicker(c, "Submission contract")
    title(c, "What the scoring harness runs")

    card(c, MARGIN, H - 320, W - 2 * MARGIN, 200, accent=TEAL)
    lines = [
        ("Input", "/input/tasks.json  →  [{task_id, video_url, styles}]"),
        ("Output", "/output/results.json  →  [{task_id, captions{…}}]"),
        ("Image", "ghcr.io/skx56/sevcap-grounded:latest"),
        ("Platform", "linux/amd64 · argument-free ENTRYPOINT · FIREWORKS_API_KEY at runtime"),
        ("Defaults", "SEVCAP_CANDIDATES=3 · SEVCAP_POLISH=1 · SEVCAP_AUDIO=0 · Kimi K2.6"),
    ]
    y = H - 155
    for k, v in lines:
        c.setFillColor(TEAL_DK)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(MARGIN + 24, y, k)
        c.setFillColor(INK)
        c.setFont("Helvetica", 12)
        c.drawString(MARGIN + 120, y, v)
        y -= 28

    wrap(
        c,
        "Placeholders are written immediately; every expected task_id is finalized. "
        "Timeouts may degrade quality but must not produce MISSING_TASKS or OUTPUT_MISSING.",
        MARGIN, 120, W - 2 * MARGIN, size=12, leading=17, color=MUTE,
    )
    footer(c, 6)


def slide_results(c: canvas.Canvas) -> None:
    bg(c)
    kicker(c, "Evidence")
    title(c, "Official sample-style clips")

    # big number
    card(c, MARGIN, H - 290, 280, 170, accent=TEAL)
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 56)
    c.drawCentredString(MARGIN + 140, H - 200, "0.966")
    c.setFillColor(MUTE)
    c.setFont("Helvetica", 12)
    c.drawCentredString(MARGIN + 140, H - 230, "combined  (acc+tone)/10")
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(MARGIN + 140, H - 260, "mean acc 4.75 · tone 4.91")

    right = [
        ("8 / 8", "tasks sev-verified"),
        ("Harness OK", "schema + task coverage"),
        ("Docker OK", "ghcr.io/skx56/sevcap-grounded"),
    ]
    y = H - 145
    for head, body in right:
        card(c, MARGIN + 300, y - 50, W - 2 * MARGIN - 300, 70)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(MARGIN + 320, y - 10, head)
        c.setFillColor(MUTE)
        c.setFont("Helvetica", 12)
        c.drawString(MARGIN + 320, y - 32, body)
        y -= 85

    wrap(
        c,
        "Internal vision judge mirrors Track 2 axes on the eight AMD sample clips "
        "(boulevard, kitten, office, mountains, waves, intersection, cooking, track).",
        MARGIN, 100, W - 2 * MARGIN, size=12, leading=16, color=MUTE,
    )
    footer(c, 7)


def slide_close(c: canvas.Canvas) -> None:
    bg(c)
    kicker(c, "Ship it")
    title(c, "Stack & links")

    left = [
        "Kimi K2.6 on Fireworks (VLM + text)",
        "Python 3.11 · asyncio · ffmpeg",
        "Docker linux/amd64 → GHCR",
        "Optional Gemma override for bonus chase",
    ]
    y = H - 150
    for item in left:
        c.setFillColor(TEAL)
        c.circle(MARGIN + 5, y + 4, 4, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica", 13)
        c.drawString(MARGIN + 18, y, item)
        y -= 32

    card(c, MARGIN + 380, H - 320, W - 2 * MARGIN - 380, 200, accent=TEAL)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN + 400, H - 160, "Submit this image")
    c.setFillColor(TEAL_DK)
    c.setFont("Helvetica-Bold", 11)
    wrap(
        c,
        "ghcr.io/skx56/sevcap-grounded:latest",
        MARGIN + 400, H - 190, W - 2 * MARGIN - 420, size=12, leading=16, color=TEAL_DK,
    )
    c.setFillColor(MUTE)
    c.setFont("Helvetica", 11)
    wrap(
        c,
        "Repo: github.com/skx56/sev-cap\n"
        "Key: FIREWORKS_API_KEY only at runtime\n"
        "Never baked into the image",
        MARGIN + 400, H - 230, W - 2 * MARGIN - 420, size=11, leading=16, color=MUTE,
    )
    footer(c, 8)


def main() -> None:
    c = canvas.Canvas(str(OUT), pagesize=landscape(A4))
    for slide in (
        slide_title,
        slide_problem,
        slide_approach,
        slide_pipeline,
        slide_styles,
        slide_harness,
        slide_results,
        slide_close,
    ):
        slide(c)
        c.showPage()
    c.save()
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
