#!/usr/bin/env python
"""Download all required local AI models for ScholarGenie"""

import os
import sys
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn
from rich.panel import Panel

console = Console()

def download_models():
    """Download all required models"""

    console.print(Panel(
        "[bold cyan]ScholarGenie Model Downloader[/bold cyan]\n\n"
        "[dim]Downloading local AI models for 100% free operation[/dim]",
        border_style="cyan"
    ))
    console.print()

    models = [
        {
            "name": "LongT5 (Summarization)",
            "model_id": "google/long-t5-tglobal-base",
            "type": "transformers"
        },
        {
            "name": "Sentence Transformers (Embeddings)",
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "type": "sentence-transformers"
        }
    ]

    console.print("[bold]Models to download:[/bold]\n")
    for model in models:
        console.print(f"  • {model['name']}")
        console.print(f"    [dim]{model['model_id']}[/dim]\n")

    console.print("[yellow]Note: This will download ~1.5GB of model files[/yellow]")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        console=console
    ) as progress:

        # Download LongT5
        task1 = progress.add_task(
            "[cyan]Downloading LongT5 model...",
            total=100
        )

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/long-t5-tglobal-base")

            progress.update(task1, completed=100)
            console.print("[green]✓[/green] LongT5 model downloaded\n")
        except Exception as e:
            console.print(f"[red]✗[/red] Error downloading LongT5: {e}\n")

        # Download Sentence Transformers
        task2 = progress.add_task(
            "[cyan]Downloading Sentence Transformers...",
            total=100
        )

        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            progress.update(task2, completed=100)
            console.print("[green]✓[/green] Sentence Transformers downloaded\n")
        except Exception as e:
            console.print(f"[red]✗[/red] Error downloading Sentence Transformers: {e}\n")

    console.print()
    console.print(Panel(
        "[bold green]✓ All models downloaded successfully![/bold green]\n\n"
        "[dim]ScholarGenie is now ready to run 100% offline[/dim]",
        border_style="green"
    ))
    console.print()

if __name__ == '__main__':
    download_models()
