"""sevcap CLI: run pipeline, smoke-check vision, lineup-test exemplars."""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="SEV-Cap: Semantic-Entropy Verified video captioning.")
console = Console()


@app.command()
def run(
    input_dir: str = typer.Option(None, "--input", "-i", help="Directory of video clips"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Directory for JSON results"),
):
    """Caption every clip in the input directory (the container entrypoint)."""
    from .pipeline import run as run_pipeline

    summary = asyncio.run(run_pipeline(input_dir, output_dir))
    console.print_json(json.dumps(summary))


@app.command()
def check():
    """Verify the Fireworks key works and which model accepts image input."""
    from .fireworks import Gemma

    async def _check():
        llm = Gemma()
        model = await llm.check_vision()
        return model

    model = asyncio.run(_check())
    console.print(f"[green]Vision OK[/green] via model: [bold]{model}[/bold]")


@app.command("lineup-test")
def lineup_test():
    """Blind-lineup the hand-written exemplars themselves (style QA)."""
    from .fireworks import Gemma
    from .gates import blind_lineup
    from .styles import STYLES

    async def _test():
        llm = Gemma()
        await llm.resolve_text_model()
        n_scenarios = len(next(iter(STYLES.values())).exemplars)
        table = Table(title="Exemplar blind-lineup results")
        table.add_column("Scenario")
        table.add_column("Style")
        table.add_column("Judged as")
        table.add_column("Conf")
        table.add_column("Pass")
        all_pass = True
        for i in range(n_scenarios):
            captions = {k: s.exemplars[i][1] for k, s in STYLES.items()}
            results = await blind_lineup(llm, captions, rng_seed=i)
            for k, r in results.items():
                ok = "[green]yes[/green]" if r.passed else "[red]NO[/red]"
                all_pass = all_pass and r.passed
                table.add_row(str(i + 1), k, r.judged_as, str(r.confidence), ok)
        console.print(table)
        if not all_pass:
            raise typer.Exit(1)

    asyncio.run(_test())


@app.command()
def facts(
    video: str = typer.Argument(..., help="Path to one video clip"),
    k: int = typer.Option(None, "--k", help="Number of extraction samples"),
):
    """Debug: run Stage 1 + semantic-entropy verification on one clip."""
    from .config import settings
    from .entropy import verify_facts
    from .extractor import extract_facts
    from .fireworks import Gemma
    from .sampler import sample_keyframes

    async def _facts():
        llm = Gemma()
        await llm.resolve_text_model()
        frames = sample_keyframes(video, settings.n_frames)
        console.print(f"Sampled {len(frames)} keyframes")
        extractions = await extract_facts(llm, frames, k=k or settings.k_samples)
        sheet = await verify_facts(llm, extractions, settings.min_support)
        console.print_json(json.dumps(sheet.report()))

    asyncio.run(_facts())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
