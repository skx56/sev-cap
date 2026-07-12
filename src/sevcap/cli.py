"""sevcap CLI: run pipeline and smoke-check vision."""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console

app = typer.Typer(help="SEV-Cap: grounded multi-style video captioning.")
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
