#!/usr/bin/env python3
"""Generate an 8-page SEV-Cap project presentation PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
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
OUT = ROOT / "docs" / "SEV-Cap-Presentation.pdf"

# Brand palette
BG = colors.HexColor("#0b0b14")
ACCENT = colors.HexColor("#8b5cf6")
PINK = colors.HexColor("#ec4899")
CYAN = colors.HexColor("#22d3ee")
AMBER = colors.HexColor("#f59e0b")
TEXT = colors.HexColor("#e7e7f0")
MUTED = colors.HexColor("#9a9ab0")
CARD = colors.HexColor("#15151f")


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=34,
            leading=40,
            textColor=ACCENT,
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=14,
            leading=18,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=ACCENT,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=PINK,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=TEXT,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=14,
            textColor=TEXT,
            leftIndent=14,
            bulletIndent=0,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "link": ParagraphStyle(
            "link",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=CYAN,
            alignment=TA_CENTER,
        ),
    }


class DarkPageCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_states = []

    def showPage(self):
        self._draw_background()
        self._draw_footer()
        super().showPage()

    def save(self):
        self._draw_background()
        self._draw_footer()
        super().save()

    def _draw_background(self):
        self.saveState()
        self.setFillColor(BG)
        self.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        # soft glow accents
        self.setFillColor(colors.Color(0.55, 0.36, 0.96, alpha=0.08))
        self.circle(40, A4[1] - 40, 120, fill=1, stroke=0)
        self.setFillColor(colors.Color(0.93, 0.28, 0.60, alpha=0.06))
        self.circle(A4[0] - 30, A4[1] - 80, 100, fill=1, stroke=0)
        self.restoreState()

    def _draw_footer(self):
        self.saveState()
        self.setStrokeColor(colors.Color(1, 1, 1, alpha=0.08))
        self.setLineWidth(0.5)
        self.line(2 * cm, 1.6 * cm, A4[0] - 2 * cm, 1.6 * cm)
        self.setFont("Helvetica", 8)
        self.setFillColor(MUTED)
        self.drawString(2 * cm, 1.1 * cm, "SEV-Cap — Semantic-Entropy Verified Video Captioning")
        self.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {self._pageNumber}")
        self.restoreState()


def _bullets(st, items: list[str]) -> list:
    out = []
    for item in items:
        out.append(Paragraph(f"• {item}", st["bullet"]))
    return out


def _style_table(st) -> Table:
    data = [
        [Paragraph("<b>Formal</b>", st["body"]), Paragraph("Clear, professional scene description", st["body"])],
        [Paragraph("<b>Sarcastic</b>", st["body"]), Paragraph("Dry, witty take with attitude", st["body"])],
        [Paragraph("<b>Humorous · Tech</b>", st["body"]), Paragraph("Playful joke with a nerdy twist", st["body"])],
        [Paragraph("<b>Humorous · Non-Tech</b>", st["body"]), Paragraph("Light humor anyone can follow", st["body"])],
    ]
    t = Table(data, colWidths=[4.2 * cm, 11.5 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CARD),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.Color(1, 1, 1, alpha=0.12)),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.Color(1, 1, 1, alpha=0.08)),
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
                ("BOX", (0, 0), (-1, -1), 0.5, colors.Color(1, 1, 1, alpha=0.12)),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.Color(1, 1, 1, alpha=0.08)),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return t


def build() -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    st = _styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.4 * cm,
    )

    story: list = []

    # Page 1 — Title
    story.append(Spacer(1, 2.8 * cm))
    story.append(Paragraph("SEV-Cap", st["title"]))
    story.append(Paragraph("Semantic-Entropy Verified Video Captioning", st["subtitle"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(
        Paragraph(
            "Hackathon Track 2 · Video Captioning<br/>"
            "Four styles · Verified facts · Anytime output",
            st["subtitle"],
        )
    )
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("Live demo", st["h2"]))
    story.append(
        Paragraph(
            '<link href="https://associates-desired-avatar-doctrine.trycloudflare.com">'
            "associates-desired-avatar-doctrine.trycloudflare.com</link>",
            st["link"],
        )
    )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("github.com/skx56/sev-cap", st["small"]))
    story.append(Paragraph("ghcr.io/skx56/sev-cap:latest", st["small"]))
    story.append(PageBreak())

    # Page 2 — Problem
    story.append(Spacer(1, 0.2 * cm))
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
                "<b>Tone:</b> a caption labeled “sarcastic” often reads as plain "
                "description — the style label is aspirational, not earned.",
            ],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Why it matters", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "The judge scores each caption on accuracy (1–5) and tone (1–5).",
                "A pipeline that cannot detect its own hallucinations or style misses "
                "leaves points on the table every clip.",
                "Hackathon containers also time out — missing output is a zero.",
            ],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "<b>SEV-Cap</b> attacks both failure modes structurally — not with a "
            "bigger prompt, but with verification gates the judge actually cares about.",
            st["body"],
        )
    )
    story.append(PageBreak())

    # Page 3 — Four outputs + overview
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Four Required Outputs", st["h1"]))
    story.append(
        Paragraph(
            "Every clip produces exactly four captions — one per mandated style:",
            st["body"],
        )
    )
    story.append(_style_table(st))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Two-phase pipeline", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "<b>Phase 1 — Draft:</b> fast vision pass writes all four styles "
                "immediately (anytime guarantee).",
                "<b>Phase 2 — SEV upgrade:</b> sample facts, verify with semantic "
                "entropy, regenerate with gates, keep draft unless both gates pass.",
            ],
        )
    )
    story.append(PageBreak())

    # Page 4 — Stage 1
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Stage 1 — Verified Facts", st["h1"]))
    story.append(Paragraph("Evidence collection", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "<b>Keyframes:</b> ffmpeg samples uniform + scene-change frames.",
                "<b>Audio:</b> faster-whisper transcribes locally (CPU, baked into Docker).",
                "Dialogue and sound cues become facts vision alone would miss.",
            ],
        )
    )
    story.append(Paragraph("Semantic-entropy verification", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "K=3 independent vision extractions (frames + transcript).",
                "Atomic facts clustered by bidirectional entailment (LLM as NLI judge).",
                "Per-fact support across samples → semantic-entropy signal.",
                "Facts in ≥2 of 3 samples are <b>verified</b>; singletons are rejected "
                "(Farquhar et al., Nature 2024 confabulation signature).",
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

    # Page 5 — Stage 2 + gates
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Stage 2 — Grounded Generation", st["h1"]))
    story.append(
        Paragraph(
            "Four style generators rewrite captions from the verified fact sheet "
            "<b>and</b> keyframes — vision context prevents fact-sheet-only captions "
            "from missing story beats.",
            st["body"],
        )
    )
    story.append(Paragraph("Gate A — Grounding", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "Every concrete claim in a caption must be entailed by the fact sheet.",
                "Failures trigger Self-Refine with the offending claims as feedback.",
            ],
        )
    )
    story.append(Paragraph("Gate B — Blind style lineup", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "All four captions are label-stripped, shuffled, and re-identified "
                "by a fresh judge (confidence ≥ 3/5).",
                "A “sarcastic” caption that reads declarative loses and is rewritten.",
                "One Self-Refine round (Madaan et al., NeurIPS 2023).",
            ],
        )
    )
    story.append(
        Paragraph(
            "<b>Draft firewall:</b> upgraded captions replace the draft only when "
            "they pass <i>both</i> gates. Otherwise the reliable vision draft is kept.",
            st["body"],
        )
    )
    story.append(PageBreak())

    # Page 6 — Robustness
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Built for the Scoring Harness", st["h1"]))
    story.append(Paragraph("Anytime algorithm", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "Phase 1 writes valid JSON for every clip before Phase 2 starts.",
                "Upgrades overwrite atomically — timeouts degrade quality, never "
                "produce missing output.",
                "Per-clip upgrade timeout (default 300s) + global time budget (1800s).",
            ],
        )
    )
    story.append(Paragraph("Container", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "linux/amd64 Docker image on ghcr.io — argument-free entrypoint.",
                "Whisper weights baked at build time (no extra network at runtime).",
                "FIREWORKS_API_KEY injected by harness; never baked into image.",
                "CI: pytest on push → build & publish via GitHub Actions.",
            ],
        )
    )
    story.append(Paragraph("Model strategy", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "<b>Default:</b> Kimi K2.6 (Fireworks serverless) — extractor, judge, "
                "writer, refiner.",
                "<b>Bonus mode:</b> Gemma 4 26B on-demand via env override, Kimi fallback "
                "on cold start or degenerate output.",
            ],
        )
    )
    story.append(PageBreak())

    # Page 7 — Architecture (text diagram)
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Architecture", st["h1"]))
    arch = """
    <font face="Courier" size="9" color="#c4b5fd">
    Video clip<br/>
      ├─ ffmpeg ──────────► keyframes<br/>
      └─ faster-whisper ──► transcript<br/>
              │<br/>
              ├─► Phase 1: vision draft (4 styles) ──► atomic write<br/>
              │<br/>
              └─► Phase 2: K=3 extractions<br/>
                    └─► entailment clustering<br/>
                          └─► semantic entropy (≥2/3 support)<br/>
                                └─► verified fact sheet<br/>
                                      └─► 4 style generators (+ keyframes)<br/>
                                            ├─ Gate A: grounding<br/>
                                            ├─ Gate B: blind lineup<br/>
                                            └─ Self-Refine (1 round)<br/>
                                                  ├─ both pass → upgraded output<br/>
                                                  └─ fail/timeout → keep draft<br/>
    </font>
    """
    story.append(Paragraph(arch, st["body"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "Every clip JSON ships a verification report: verified facts, rejected "
            "high-entropy facts, lineup verdicts, retry history.",
            st["body"],
        )
    )
    story.append(PageBreak())

    # Page 8 — Results + stack + CTA
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Results & Stack", st["h1"]))
    story.append(Paragraph("Internal eval (7 clips, LLM-Judge)", st["h2"]))
    story.append(_results_table(st))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Tech stack", st["h2"]))
    story.extend(
        _bullets(
            st,
            [
                "Kimi K2.6 · Gemma 4 26B (bonus) · Fireworks AI API",
                "faster-whisper · ffmpeg · Python 3.11 · asyncio",
                "Streamlit demo · Docker linux/amd64 · GitHub Actions → ghcr.io",
            ],
        )
    )
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.Color(1, 1, 1, alpha=0.12)))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Try it live", st["h2"]))
    story.append(
        Paragraph(
            '<link href="https://associates-desired-avatar-doctrine.trycloudflare.com">'
            "associates-desired-avatar-doctrine.trycloudflare.com</link>",
            st["link"],
        )
    )
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            "Upload a clip · get four verified captions · inspect the fact report",
            st["small"],
        )
    )

    doc.build(story, canvasmaker=DarkPageCanvas)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
