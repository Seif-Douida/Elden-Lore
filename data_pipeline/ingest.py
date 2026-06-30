"""
data_pipeline/ingest.py

Phase 4 (final) of the pipeline: INGEST.

Reads chunks.jsonl, embeds each chunk with bge-base-en-v1.5 (768-dim, local,
free), and upserts the vectors + metadata payloads into a local Qdrant
collection. After this runs, the corpus is a live, queryable vector DB.

Retrieval design:
  - Dense vectors (bge-base) drive semantic search.
  - The entities[] payload (from the chunker's gazetteer) provides the
    exact proper-noun matching axis — so "what does Malenia drop" can filter
    to chunks tagged with 'Malenia' alongside the vector search.
  - Payload indexes on category / chunk_type / doc_type / entities make those
    filters fast.

bge note: passages are embedded as-is. The matching QUERY-side instruction
("Represent this sentence for searching relevant passages:") is applied later
in the retriever, NOT here.

Qdrant connection is read from env (QDRANT_HOST / QDRANT_PORT), default
localhost:6333 — matching the docker-compose service.

Usage:
    uv run python data_pipeline/ingest.py                 # create + ingest
    uv run python data_pipeline/ingest.py --recreate      # drop & rebuild collection
    uv run python data_pipeline/ingest.py --limit 100     # test slice
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from dotenv import load_dotenv

from common import DATA_DIR

load_dotenv()
console = Console()

# ── Configuration ─────────────────────────────────────────────────────────────

CHUNKS_FILE     = DATA_DIR / "chunks.jsonl"
EMBED_MODEL     = "BAAI/bge-base-en-v1.5"
VECTOR_DIM      = 768
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "elden_ring")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

EMBED_BATCH  = int(os.getenv("EMBED_BATCH", "64"))    # encode batch (GPU-tuned)
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "256"))   # points per Qdrant upsert

# Device selection: CUDA if available, else CPU. Override with EMBED_DEVICE=cpu.
def _select_device() -> str:
    forced = os.getenv("EMBED_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ── Point id ──────────────────────────────────────────────────────────────────

def point_id(chunk_id: str) -> int:
    """
    chunk_id is a 16-char hex string (64 bits) → a valid uint64 Qdrant point id.
    Deterministic, so re-ingesting overwrites rather than duplicating.
    """
    return int(chunk_id, 16)


# ── Load chunks ───────────────────────────────────────────────────────────────

def load_chunks(limit: Optional[int]) -> list[dict]:
    if not CHUNKS_FILE.exists():
        console.print(f"[red]No chunks file at {CHUNKS_FILE}. Run chunker.py first.[/red]")
        return []
    rows = [json.loads(l) for l in CHUNKS_FILE.open(encoding="utf-8")]
    return rows[:limit] if limit else rows


# ── Main ──────────────────────────────────────────────────────────────────────

def run(limit: Optional[int] = None, recreate: bool = False) -> None:
    chunks = load_chunks(limit)
    if not chunks:
        return
    console.print(f"[bold cyan]Ingesting {len(chunks)} chunks[/bold cyan]")

    # Imports deferred so --help is instant
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        PayloadSchemaType,
    )

    # 1. Embedding model — on GPU if available
    device = _select_device()
    console.print(f"[dim]Loading embedder: {EMBED_MODEL} on [bold]{device}[/bold][/dim]")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    if device == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            console.print(f"[dim]GPU: {name} ({vram:.1f} GB) · batch={EMBED_BATCH}[/dim]")
        except Exception:
            pass

    # 2. Qdrant client
    console.print(f"[dim]Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}[/dim]")
    client = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
        https=False,                 # local Qdrant speaks plain HTTP, not TLS
        prefer_grpc=False,
        check_compatibility=False,   # skip the version-probe warning
    )

    # 3. Collection
    exists = client.collection_exists(COLLECTION_NAME)
    if exists and recreate:
        console.print(f"[yellow]Dropping existing collection '{COLLECTION_NAME}'[/yellow]")
        client.delete_collection(COLLECTION_NAME)
        exists = False
    if not exists:
        console.print(f"Creating collection '{COLLECTION_NAME}' (dim={VECTOR_DIM}, cosine)")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        # Payload indexes for fast filtering
        for field, schema in [
            ("category",   PayloadSchemaType.KEYWORD),
            ("chunk_type", PayloadSchemaType.KEYWORD),
            ("doc_type",   PayloadSchemaType.KEYWORD),
            ("entities",   PayloadSchemaType.KEYWORD),
            ("breadcrumb", PayloadSchemaType.KEYWORD),
            ("title",      PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
        console.print("[dim]Payload indexes created: category, chunk_type, doc_type, entities, breadcrumb, title[/dim]")
    else:
        console.print(f"[dim]Collection '{COLLECTION_NAME}' exists — upserting (use --recreate to rebuild)[/dim]")

    # 4. Embed + upsert in batches
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("Embedding + upserting...", total=len(chunks))

        buffer: list[PointStruct] = []
        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start:start + EMBED_BATCH]
            texts = [c["text"] for c in batch]

            # bge passages embedded as-is; normalize for cosine
            vectors = model.encode(
                texts, batch_size=EMBED_BATCH,
                normalize_embeddings=True, show_progress_bar=False,
            )

            for chunk, vec in zip(batch, vectors):
                payload = {
                    "text":            chunk["text"],
                    "raw_text":        chunk["raw_text"],
                    "url":             chunk["url"],
                    "title":           chunk["title"],
                    "category":        chunk["category"],
                    "doc_type":        chunk["doc_type"],
                    "breadcrumb":      chunk["breadcrumb"],
                    "section_heading": chunk["section_heading"],
                    "chunk_type":      chunk["chunk_type"],
                    "entities":        chunk["entities"],
                    "image_url":       chunk["image_url"],
                    "source_type":     chunk.get("source_type", "wiki"),
                }
                buffer.append(PointStruct(
                    id=point_id(chunk["chunk_id"]),
                    vector=vec.tolist(),
                    payload=payload,
                ))

            progress.update(task, advance=len(batch))

            if len(buffer) >= UPSERT_BATCH:
                client.upsert(COLLECTION_NAME, points=buffer, wait=False)
                buffer = []

        if buffer:
            client.upsert(COLLECTION_NAME, points=buffer, wait=True)

    # 5. Durability: wait for the optimizer to finish so the on-disk segments
    # are fully written BEFORE the script exits. This is what prevents the
    # "stop the container while it's still optimizing → truncated payload file"
    # corruption we hit last time. We poll until indexed == total (or green).
    import time
    console.print("[dim]Waiting for Qdrant to finish indexing (do not stop the container yet)…[/dim]")
    target = len(chunks)
    for _ in range(120):                      # up to ~4 minutes
        info = client.get_collection(COLLECTION_NAME)
        indexed = info.indexed_vectors_count or 0
        status = info.status
        if status == "green" and indexed >= target:
            break
        time.sleep(2)

    # Snapshot so there's a durable backup that survives any future mishap.
    try:
        snap = client.create_snapshot(collection_name=COLLECTION_NAME, wait=True)
        snap_name = getattr(snap, "name", str(snap))
        console.print(f"[dim]Snapshot created: {snap_name}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Snapshot skipped: {e}[/yellow]")

    # 6. Report
    info = client.get_collection(COLLECTION_NAME)
    console.print(f"\n[bold green]Done![/bold green] Collection '{COLLECTION_NAME}'")
    console.print(f"  status: {info.status}")
    console.print(f"  points: {info.points_count}")
    console.print(f"  indexed vectors: {info.indexed_vectors_count}")
    console.print(f"  vector dim: {VECTOR_DIM}, distance: cosine")
    console.print(f"  Qdrant dashboard: http://{QDRANT_HOST}:{QDRANT_PORT}/dashboard")
    if (info.indexed_vectors_count or 0) < info.points_count:
        console.print(
            "  [yellow]note: indexing still catching up — wait until "
            "indexed == points before stopping the container.[/yellow]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Elden Ring wiki — Qdrant ingest")
    parser.add_argument("--limit", type=int, default=None,
                        help="Ingest only the first N chunks (testing)")
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and rebuild the collection from scratch")
    args = parser.parse_args()
    run(limit=args.limit, recreate=args.recreate)


if __name__ == "__main__":
    main()