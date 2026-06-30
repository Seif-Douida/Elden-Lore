"""
data_pipeline/upload.py

Stage 4b of the pipeline: UPLOAD (decoupled from embedding).

Loads the pre-computed vectors + ids + payloads written by embed.py and upserts
them into Qdrant. No GPU, no model — pure I/O, so it's fast and fully retryable.
If Qdrant ever corrupts, re-run this in seconds instead of re-embedding.

Hardening against the payload-gridstore corruption we hit twice:
  - single optimization thread (shrinks the concurrent-write window)
  - upload with wait, then poll until status==green AND indexed==points, with a
    long ceiling — the script does NOT exit while the optimizer is still writing
  - snapshot only AFTER green, so a durable backup exists

Usage:
    uv run python data_pipeline/upload.py --recreate
    uv run python data_pipeline/upload.py            # upsert into existing
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from dotenv import load_dotenv

from common import DATA_DIR

load_dotenv()
console = Console()

# ── Configuration ─────────────────────────────────────────────────────────────

EMBED_DIR     = DATA_DIR / "embeddings"
VECTORS_FILE  = EMBED_DIR / "vectors.npy"
IDS_FILE      = EMBED_DIR / "ids.npy"
PAYLOADS_FILE = EMBED_DIR / "payloads.jsonl"
META_FILE     = EMBED_DIR / "meta.json"

VECTOR_DIM      = 768
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "elden_ring")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "256"))
WAIT_CEILING_SEC = int(os.getenv("WAIT_CEILING_SEC", "1200"))   # 20 min


def _load_artifacts():
    if not VECTORS_FILE.exists():
        console.print(f"[red]No embeddings at {EMBED_DIR}. Run embed.py first.[/red]")
        return None
    vectors = np.load(VECTORS_FILE)
    ids = np.load(IDS_FILE)
    payloads = [json.loads(l) for l in PAYLOADS_FILE.open(encoding="utf-8")]
    if not (len(vectors) == len(ids) == len(payloads)):
        console.print(f"[red]Length mismatch: vectors={len(vectors)} ids={len(ids)} "
                      f"payloads={len(payloads)}[/red]")
        return None
    if META_FILE.exists():
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        if not meta.get("ok", True):
            console.print("[yellow]Warning: embed.py reported failing sanity checks.[/yellow]")
    return vectors, ids, payloads


def _wait_for_green(client, target: int) -> None:
    console.print("[dim]Waiting for optimizer to finish (status=green, indexed==points). "
                  "Do NOT stop the container.[/dim]")
    start = time.time()
    while time.time() - start < WAIT_CEILING_SEC:
        info = client.get_collection(COLLECTION_NAME)
        status = info.status
        indexed = info.indexed_vectors_count or 0
        elapsed = int(time.time() - start)
        console.print(f"  [dim]{elapsed:4d}s  status={status}  indexed={indexed}/{target}[/dim]")
        if str(status) == "green" and indexed >= target:
            console.print("[green]Optimizer settled — green and fully indexed.[/green]")
            return
        time.sleep(10)
    console.print("[yellow]Wait ceiling reached; optimizer may still be working. "
                  "Check before stopping the container.[/yellow]")


def run(recreate: bool = False) -> None:
    art = _load_artifacts()
    if art is None:
        return
    vectors, ids, payloads = art
    n = len(vectors)
    console.print(f"[bold cyan]Uploading {n} pre-computed vectors[/bold cyan]")

    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, PayloadSchemaType, OptimizersConfigDiff,
    )

    client = QdrantClient(
        host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY,
        https=False, check_compatibility=False,
    )

    exists = client.collection_exists(COLLECTION_NAME)
    if exists and recreate:
        console.print(f"[yellow]Dropping existing collection '{COLLECTION_NAME}'[/yellow]")
        client.delete_collection(COLLECTION_NAME)
        exists = False
    if not exists:
        console.print(f"Creating '{COLLECTION_NAME}' (dim={VECTOR_DIM}, cosine, "
                      f"single-thread optimizer)")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            # Hardening: one optimization thread = minimal concurrent-write window.
            optimizers_config=OptimizersConfigDiff(max_optimization_threads=1),
        )
        for field, schema in [
            ("category",   PayloadSchemaType.KEYWORD),
            ("chunk_type", PayloadSchemaType.KEYWORD),
            ("doc_type",   PayloadSchemaType.KEYWORD),
            ("entities",   PayloadSchemaType.KEYWORD),
            ("breadcrumb", PayloadSchemaType.KEYWORD),
            ("title",      PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
        console.print("[dim]Payload indexes created.[/dim]")

    # Upsert in batches (pure I/O)
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("Upserting…", total=n)
        for start in range(0, n, UPSERT_BATCH):
            end = min(start + UPSERT_BATCH, n)
            points = [
                PointStruct(
                    id=int(ids[i]),
                    vector=vectors[i].tolist(),
                    payload=payloads[i],
                )
                for i in range(start, end)
            ]
            # wait=True on every batch: each write is durably acknowledged before
            # the next. Slower than fire-and-forget, but no silent backlog.
            client.upsert(COLLECTION_NAME, points=points, wait=True)
            progress.update(task, advance=end - start)

    # Hold until the optimizer is fully settled before exiting.
    _wait_for_green(client, n)

    # Snapshot only after green — a durable restore point.
    try:
        snap = client.create_snapshot(collection_name=COLLECTION_NAME, wait=True)
        console.print(f"[dim]Snapshot: {getattr(snap, 'name', snap)}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Snapshot skipped: {e}[/yellow]")

    info = client.get_collection(COLLECTION_NAME)
    console.print(f"\n[bold green]Done![/bold green] '{COLLECTION_NAME}'")
    console.print(f"  status: {info.status}")
    console.print(f"  points: {info.points_count}")
    console.print(f"  indexed vectors: {info.indexed_vectors_count}")
    console.print(f"  dashboard: http://{QDRANT_HOST}:{QDRANT_PORT}/dashboard")


def main() -> None:
    parser = argparse.ArgumentParser(description="Elden Ring — upload embeddings to Qdrant")
    parser.add_argument("--recreate", action="store_true", help="drop & rebuild the collection")
    args = parser.parse_args()
    run(recreate=args.recreate)


if __name__ == "__main__":
    main()