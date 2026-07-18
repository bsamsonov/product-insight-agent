"""CLI to embed and index documents into Qdrant."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

app = typer.Typer(help="Index documents into Qdrant vector store.")
_log = logging.getLogger(__name__)


@app.command()
def main(
    source: Path = typer.Option(..., help="Path to JSONL source file"),  # noqa: B008
    tenant: str = typer.Option("default", help="Tenant name"),
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant URL"),
    limit: int | None = typer.Option(None, help="Max documents to index"),
    batch_size: int = typer.Option(64, help="Embedding batch size"),
    recreate: bool = typer.Option(
        False, "--recreate", help="Drop and recreate the collection before indexing"
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Skip chunks already present in the collection (skips re-embedding them), so an "
        "interrupted run can continue. On by default; ignored when --recreate is set. "
        "Pass --no-resume to force re-embedding of every chunk.",
    ),
) -> None:
    logging.basicConfig(level=logging.INFO)

    # --recreate starts from an empty collection, so resume would only add a useless
    # existence check per batch; disable it in that case.
    skip_existing = resume and not recreate

    from poc.ingestion.chunker import RecursiveTokenChunker
    from poc.ingestion.normalizer import normalize
    from poc.ingestion.sources.jsonl import JsonlSource
    from poc.retrieval.bge_embedder import BgeM3Embedder
    from poc.retrieval.qdrant_index import QdrantIndex

    _log.info("Loading embedder...")
    embedder = BgeM3Embedder()

    _log.info("Connecting to Qdrant at %s for tenant '%s'...", qdrant_url, tenant)
    index = QdrantIndex(embedder=embedder, qdrant_url=qdrant_url, tenant=tenant, recreate=recreate)

    chunker = RecursiveTokenChunker()
    src = JsonlSource(source)

    total_chunks = 0
    all_chunks = []

    for i, raw_doc in enumerate(src.iter()):
        if limit and i >= limit:
            break
        doc = normalize(raw_doc)
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)

        if len(all_chunks) >= batch_size:
            index.upsert(all_chunks, batch_size=batch_size, skip_existing=skip_existing)
            total_chunks += len(all_chunks)
            all_chunks = []

    if all_chunks:
        index.upsert(all_chunks, batch_size=batch_size)
        total_chunks += len(all_chunks)

    _log.info("Indexed %d chunks into collection 'reviews__%s'", total_chunks, tenant)
    typer.echo(f"Done: {total_chunks} chunks indexed.")


if __name__ == "__main__":
    app()
