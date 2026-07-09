#!/usr/bin/env python3
"""Generate an 8-page SEV-Cap project presentation PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATHS = [
    ROOT / "docs" / "SEV-Cap-Presentation.pdf",
    ROOT / "SEV-Cap-Presentation.pdf",
]

# Palette — light background, dark readable text
ACCENT = colors.HexColor("#6d28d9")
PINK = colors.HexColor("#be185d")
CYAN = colors.HexColor("#0e7490")
TEXT = colors.HexColor("#1e1e2e")
MUTED = colors.HexColor("#4b5563")
LIGHT_BG = colors.HexColor("#f8f7ff")
CARD = colors.HexColor("#ede9fe")
BORDER = colors.HexColor("#c4b5fd")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            fontName="Helvetica-Bold",
            fontSize=36,
            leading=42,
            textColor=ACCENT,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Helvetica",
            fontSize=14,
            leading=18,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1",
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=ACCENT,
            spaceAfter=12,
        ),
        "h2": ParagraphStyle(
            "h2",
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=PINK,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=TEXT,
            spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=TEXT,
            leftIndent=16,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "link": ParagraphStyle(
            "link",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=CYAN,
            alignment=TA_CENTER,
        ),
        "mono": ParagraphStyle(
            "mono",
            fontName="Courier",
            fontSize=8.5,
            leading=11,
            textColor=ACCENT,
            spaceAfter=4,
        ),
    }


def _page_decor(canvas, doc):
    """Footer + top accent bar. Called after page content — only touches margins."""
    canvas.saveState()
    w, h = A4
    # top accent stripe
    canvas.setFillColor(ACCENT)
    canvas.rect(0, h - 6, w, 6, fill=1, stroke=0)
    # footer line
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, 1.5 * cm, w - 2 * cm, 1.5 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.0 * cm, "SEV-Cap — Semantic-Entropy Verified Video Captioning")
    canvas.drawRightString(w - 2 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _bullets(st, items: list[str]) -> list:
    return [Paragraph(f"&bull; {item}", st["bullet"]) for item in items]


def _style_table(st) -> Table:
    data = [
        [Paragraph("<b>Formal</b>", st["body"]), Paragraph("Clear, professional scene description", st["body"])],
        [Paragraph("<b>Sarcastic</b>", st["body"]), Paragraph("Dry, witty take with a bit of attitude", st["body"])],
        [Paragraph("<b>Humorous · Tech</b>", st["body"]), Paragraph("Playful joke with a nerdy / tech twist", st["body"])],
        [Paragraph("<b>Humorous · Non-Tech</b>", st["body"]), Paragraph("Light, everyday humor anyone can follow", st["body"])],
    ]
    t = Table(data, colWidths=[4.5 * cm, 11 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CARD),
                ("BOX", (0, 0), (-1, -1), 1, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TEXTCOLOR", (0, 0), (0, -1), ACCENT),
            ]
        )
    )
    return t


def _results_table(st) -> Table:
    data = [
        [Paragraph("<b>Metric</b>", st["body"]), Paragraph("<b>Score</b>", st["body"])],
        [Paragraph("Mean accuracy (1–5)", st["body"]), Paragraph("3.82", st["body"])],
        [Paragraph("Mean tone (1–5)", st["body"]), Paragraph("4.79", st["body"])],
        [Paragraph("Combined (leaderboard 0–1)", st["body"]), Paragraph("<b>0.86</b>", st["body"])],
    ]
    t = Table(data, colWidths=[8 * cm, 4 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), CARD),
                ("BOX", (0, 0), (-1, -1), 1, BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def _build_story(st) -> list:
    story: list = []

    # ── Page 1: Title
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("SEV-Cap", st["title"]))
    story.append(Paragraph("Semantic-Entropy Verified Video Captioning", st["subtitle"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(
        Paragraph(
            "Hackathon Track 2 · Video Captioning<br/>"
            "Four styles · Verified facts · Anytime output",
            st["subtitle"],
        )
    )
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("Live Demo", st["h2"]))
    story.append(
        Paragraph(
            '<link href="https://associates-desired-avatar-doctrine.trycloudflare.com">'
            "associates-desired-avatar-doctrine.trycloudflare.com</link>",
            st["link"],
        )
    )
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("github.com/skx56/sev-cap", st["small"]))
    story.append(Paragraph("Docker: ghcr.io/skx56/sev-cap:latest", st["small"]))
    story.append(PageBreak())

    # ── Page 2: Problem
    story.append(Paragraph("The Problem", st["h1"]))
    story.append(
        Paragraph(
            "Single-pass video captioners — one VLM call, four styles — fail the "
            "LLM-Judge on two predictable axes:",
            st["body"],
        )
    )
    story.extend(
        _bullets(
            st,
            [
                "<b>Accuracy:</b> models invent details not visible in the clip "
                "(hallucinated objects, actions, dialogue).",
                "<b>Tone:</b> a caption labeled sarcastic often reads as plain "
                "description — the style label is aspirational, not earned.",
            ],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Why it matters", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "The judge scores each caption on accuracy (1–5) and tone (1–5).",
                "Pipelines that cannot detect their own hallucinations leave points on the table.",
                "Hackathon containers time out — missing output scores zero.",
            ],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "<b>SEV-Cap</b> attacks both failure modes structurally — with verification "
            "gates the judge actually cares about, not a bigger prompt.",
            st["body"],
        )
    )
    story.append(PageBreak())

    # ── Page 3: Four outputs
    story.append(Paragraph("Four Required Outputs", st["h1"]))
    story.append(
        Paragraph(
            "Every clip produces exactly four captions — one per mandated style:",
            st["body"],
        )
    )
    story.append(_style_table(st))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Two-Phase Pipeline", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "<b>Phase 1 — Draft:</b> fast vision pass writes all four styles "
                "immediately. Every clip has valid output from second one.",
                "<b>Phase 2 — SEV Upgrade:</b> sample facts, verify with semantic "
                "entropy, regenerate with gates. Keep draft unless both gates pass.",
            ],
        )
    )
    story.append(PageBreak())

    # ── Page 4: Stage 1
    story.append(Paragraph("Stage 1 — Verified Facts", st["h1"]))
    story.append(Paragraph("Evidence Collection", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "<b>Keyframes:</b> ffmpeg samples uniform + scene-change frames.",
                "<b>Audio:</b> faster-whisper transcribes locally (CPU, baked into Docker).",
                "Dialogue and sound cues become facts that vision alone would miss.",
            ],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Semantic-Entropy Verification", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "K=3 independent vision extractions (frames + transcript).",
                "Atomic facts clustered by bidirectional entailment (LLM as NLI judge).",
                "Per-fact support across samples gives a semantic-entropy signal.",
                "Facts in 2 of 3 samples are <b>verified</b>; singletons are rejected.",
                "Based on Farquhar et al., Nature 2024 — confabulation signature.",
            ],
        )
    )
    story.append(
        Paragraph(
            "Rejected high-entropy facts are logged in every clip's verification report.",
            st["body"],
        )
    )
    story.append(PageBreak())

    # ── Page 5: Stage 2
    story.append(Paragraph("Stage 2 — Grounded Generation", st["h1"]))
    story.append(
        Paragraph(
            "Four style generators rewrite captions from the verified fact sheet "
            "<b>and</b> keyframes. Vision context prevents fact-sheet-only captions "
            "from missing story beats.",
            st["body"],
        )
    )
    story.append(Paragraph("Gate A — Grounding", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "Every concrete claim must be entailed by the verified fact sheet.",
                "Failures trigger Self-Refine with offending claims as feedback.",
            ],
        )
    )
    story.append(Paragraph("Gate B — Blind Style Lineup", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "All four captions are label-stripped, shuffled, and re-identified "
                "by a fresh judge (confidence must be 3/5 or higher).",
                "A sarcastic caption that reads declarative loses and is rewritten.",
                "One Self-Refine round (Madaan et al., NeurIPS 2023).",
            ],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "<b>Draft firewall:</b> upgraded captions replace the draft only when "
            "they pass <i>both</i> gates. Otherwise the reliable vision draft is kept.",
            st["body"],
        )
    )
    story.append(PageBreak())

    # ── Page 6: Harness
    story.append(Paragraph("Built for the Scoring Harness", st["h1"]))
    story.append(Paragraph("Anytime Algorithm", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "Phase 1 writes valid JSON for every clip before Phase 2 starts.",
                "Upgrades overwrite atomically — timeouts degrade quality, never "
                "produce missing output.",
                "Per-clip upgrade timeout (300s) + global time budget (1800s).",
            ],
        )
    )
    story.append(Paragraph("Docker Container", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "linux/amd64 image on ghcr.io — argument-free entrypoint.",
                "Whisper weights baked at build time (no extra network at runtime).",
                "FIREWORKS_API_KEY injected by harness; never baked into image.",
                "CI: pytest on push, then build and publish via GitHub Actions.",
            ],
        )
    )
    story.append(Paragraph("Model Strategy", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "<b>Default:</b> Kimi K2.6 (Fireworks serverless) — extractor, judge, "
                "writer, refiner.",
                "<b>Bonus mode:</b> Gemma 4 26B on-demand via env override, with Kimi "
                "fallback on cold start or degenerate output.",
            ],
        )
    )
    story.append(PageBreak())

    # ── Page 7: Architecture
    story.append(Paragraph("Architecture", st["h1"]))
    arch_lines = [
        "Video clip",
        "  |-- ffmpeg ------------> keyframes",
        "  |-- faster-whisper ----> transcript",
        "  |",
        "  |-- Phase 1: vision draft (4 styles) --> atomic write to disk",
        "  |",
        "  +-- Phase 2: K=3 independent extractions",
        "        |-- bidirectional entailment clustering",
        "        |-- semantic entropy (keep facts with 2/3 support)",
        "        |-- verified fact sheet",
        "        |-- 4 style generators (+ keyframes)",
        "        |     |-- Gate A: grounding check",
        "        |     |-- Gate B: blind style lineup",
        "        |     +-- Self-Refine (1 round)",
        "        |           |-- both pass --> upgraded output",
        "        +-----------+-- fail/timeout --> keep draft",
    ]
    for line in arch_lines:
        story.append(Paragraph(line.replace(" ", "&nbsp;"), st["mono"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(
        Paragraph(
            "Every clip JSON includes a verification report: verified facts, "
            "rejected high-entropy facts, lineup verdicts, and retry history.",
            st["body"],
        )
    )
    story.append(PageBreak())

    # ── Page 8: Results
    story.append(Paragraph("Results and Tech Stack", st["h1"]))
    story.append(Paragraph("Internal Eval (7 clips, LLM-Judge)", st["h2"]))
    story.append(_results_table(st))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Tech Stack", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "Kimi K2.6 (default) · Gemma 4 26B (bonus) · Fireworks AI API",
                "faster-whisper · ffmpeg · Python 3.11 · asyncio",
                "Streamlit demo · Docker linux/amd64 · GitHub Actions to ghcr.io",
            ],
        )
    )
    story.append(Spacer(1, 0.8 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Try It Live", st["h2"]))
    story.append(
        Paragraph(
            '<link href="https://associates-desired-avatar-doctrine.trycloudflare.com">'
            "associates-desired-avatar-doctrine.trycloudflare.com</link>",
            st["link"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "Upload a clip, get four verified captions, inspect the fact report.",
            st["small"],
        )
    )
    return story


def build() -> list[Path]:
    st = _styles()
    written = []
    for out in OUT_PATHS:
        out.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(out),
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2.4 * cm,
            bottomMargin=2.2 * cm,
        )
        doc.build(_build_story(st), onFirstPage=_page_decor, onLaterPages=_page_decor)
        written.append(out)
    return written


if __name__ == "__main__":
    for path in build():
        print(f"Wrote {path}")
