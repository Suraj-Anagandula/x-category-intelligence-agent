"""Index every existing `data/tweets/<category>/<date>.json` snapshot into
the Ask Intelligence vector store, so already-collected runs become
searchable without re-scraping X.

Usage:
    python scripts/backfill_rag_index.py

Shares the exact same `index_tweets` code path (and idempotency guarantee)
as live indexing - safe to run repeatedly. Requires the `rag` extra
(`pip install -e ".[rag]"` or `pip install -r requirements.txt`).
"""

from __future__ import annotations

from rich.console import Console

from app.config import settings
from app.rag.embeddings import Embedder
from app.rag.indexer import backfill_from_snapshots
from app.rag.vector_store import VectorStore


def main() -> None:
    settings.ensure_directories()
    console = Console()

    store = VectorStore(settings.chroma_dir)
    embedder = Embedder()

    console.print(f"Backfilling Ask Intelligence index from {settings.tweets_output_dir} ...")
    results = backfill_from_snapshots(settings, store, embedder)

    if not results:
        console.print("[yellow]No run snapshots found to index.[/yellow]")
        return

    total = sum(results.values())
    for run_key, count in results.items():
        console.print(f"  {run_key}: {count} tweet(s) indexed")
    console.print(f"\n[bold]Done.[/bold] {total} tweet(s) indexed across {len(results)} run(s).")
    console.print(f"Vector store: {settings.chroma_dir}")


if __name__ == "__main__":
    main()
