# SEV-Cap

<p align="center">
<img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge" />
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge" />
  <img alt="Pydantic" src="https://img.shields.io/badge/Pydantic-374151?style=for-the-badge" />
  <img alt="Docker" src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge" />
  <img alt="Machine Learning" src="https://img.shields.io/badge/Machine%20Learning-102230?style=for-the-badge" />
</p>

<p align="center">
  <strong>A semantic-entropy verified video captioning pipeline for producing multi-style captions with confidence-aware validation.</strong>
</p>

SEV-Cap focuses on one of the hardest parts of automated captioning: making generated descriptions useful without hiding uncertainty. It extracts video signals, transcribes audio, generates caption variants, and validates output through schema-backed stages.

## Core Capabilities

- Runs a modular video captioning pipeline from extraction to final caption output.
- Supports transcription, analysis, captioning, and validation stages.
- Exposes an interactive Streamlit application for experimentation.
- Includes Docker and test assets for repeatable evaluation.

## Technical Architecture

The pipeline is split into extraction, transcription, analysis, captioning, and validation modules. Pydantic schemas define data contracts between stages, while the Streamlit app provides a thin interface over the core pipeline.

## Architecture Diagram

```mermaid
%%{init: {"flowchart": {"nodeSpacing": 55, "rankSpacing": 70, "curve": "basis"}, "themeVariables": {"fontSize": "16px", "fontFamily": "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"}}}%%
flowchart TD
  Video["Video Input"] --> Extract["Frame and<br/>Audio Extraction"]
  Extract --> Transcribe["Speech Transcription"]
  Extract --> Analyze["Visual Analysis"]
  Transcribe --> Caption["Caption Generation"]
  Analyze --> Caption
  Caption --> Schemas["Pydantic Validation"]
  Schemas --> Streamlit["Streamlit Review App"]
  Schemas --> Output["Multi-Style Captions"]

  classDef inputs fill:#E0F2FE,stroke:#0284C7,color:#0C4A6E,stroke-width:2.5px;
  classDef process fill:#EDE9FE,stroke:#7C3AED,color:#4C1D95,stroke-width:2.5px;
  classDef data fill:#CCFBF1,stroke:#0D9488,color:#134E4A,stroke-width:2.5px;
  classDef agent fill:#FCE7F3,stroke:#DB2777,color:#831843,stroke-width:2.5px;
  classDef output fill:#FEF9C3,stroke:#CA8A04,color:#713F12,stroke-width:2.5px;
  class Video inputs;
  class Extract,Transcribe,Analyze,Caption,Streamlit process;
  class Schemas data;
  class Output output;
  linkStyle default stroke:#475569,stroke-width:2.5px;
```

## Technology Stack

- Python 3.11 project with pyproject configuration.
- faster-whisper for transcription workflows.
- Pydantic schemas for typed pipeline contracts.
- Streamlit UI for interactive review.
- Docker support for environment portability.

## Repository Structure

- `pipeline/extract.py` - Video extraction stage.
- `pipeline/transcribe.py` - Audio transcription stage.
- `pipeline/caption.py` - Caption generation stage.
- `schemas.py` - Typed data contracts.
- `streamlit_app.py` - Interactive app entry point.
- `tests/test_config.py` - Configuration test coverage.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
streamlit run streamlit_app.py
```

## Professional Context

This project demonstrates applied multimodal engineering, pipeline design, and validation-oriented product thinking.
