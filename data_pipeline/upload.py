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
from typing import Optional

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

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


VECTOR_DIM      = 768
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "elden_ring")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
# TLS: local Qdrant is http (even with an api key set), Qdrant Cloud is https.
# Explicit opt-in so local behaviour is unchanged; set QDRANT_HTTPS=true in cloud.
QDRANT_HTTPS = _env_bool("QDRANT_HTTPS", default=False)

UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "256"))
WAIT_CEILING_SEC = int(os.getenv("WAIT_CEILING_SEC", "1200"))   # 20 min
# Per-request timeout (seconds). Localhost is instant; a remote cluster (Qdrant
# Cloud) over a home upstream link needs headroom to send a multi-MB batch body.
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "120"))
# How many times to retry a batch on a transient network/timeout error.
UPSERT_RETRIES = int(os.getenv("UPSERT_RETRIES", "5"))


def _load_artifacts(embed_dir: Path):
    vectors_file  = embed_dir / "vectors.npy"
    ids_file      = embed_dir / "ids.npy"
    payloads_file = embed_dir / "payloads.jsonl"
    meta_file     = embed_dir / "meta.json"
    if not vectors_file.exists():
        console.print(f"[red]No embeddings at {embed_dir}. Run embed.py first.[/red]")
        return None
    vectors = np.load(vectors_file)
    ids = np.load(ids_file)
    payloads = [json.loads(l) for l in payloads_file.open(encoding="utf-8")]
    if not (len(vectors) == len(ids) == len(payloads)):
        console.print(f"[red]Length mismatch: vectors={len(vectors)} ids={len(ids)} "
                      f"payloads={len(payloads)}[/red]")
        return None
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if not meta.get("ok", True):
            console.print("[yellow]Warning: embed.py reported failing sanity checks.[/yellow]")
    return vectors, ids, payloads


def _wait_for_green(client, target: int) -> None:
    # Gate on points_count (all rows stored), NOT indexed_vectors_count: Qdrant
    # only HNSW-indexes a segment once it passes indexing_threshold (default
    # 20k), so a sub-threshold segment is searched brute-force and NEVER counted
    # as indexed — indexed_vectors_count can plateau below points_count forever.
    console.print("[dim]Waiting for status=green and all points stored. "
                  "Do NOT stop the container.[/dim]")
    start = time.time()
    while time.time() - start < WAIT_CEILING_SEC:
        info = client.get_collection(COLLECTION_NAME)
        status = info.status
        points = info.points_count or 0
        indexed = info.indexed_vectors_count or 0
        elapsed = int(time.time() - start)
        console.print(f"  [dim]{elapsed:4d}s  status={status}  points={points}/{target}  "
                      f"indexed={indexed} (sub-threshold segments stay unindexed)[/dim]")
        if str(status) == "green" and points >= target:
            console.print("[green]Settled — green and all points stored "
                          f"({points} points; {indexed} HNSW-indexed).[/green]")
            return
        time.sleep(10)
    console.print("[yellow]Wait ceiling reached; check the collection before stopping.[/yellow]")


def run(recreate: bool = False, embeddings_dir: Optional[str] = None) -> None:
    embed_dir = Path(embeddings_dir) if embeddings_dir else EMBED_DIR
    art = _load_artifacts(embed_dir)
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
        https=QDRANT_HTTPS, check_compatibility=False, timeout=QDRANT_TIMEOUT,
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

    # Ensure payload indexes (idempotent). Runs whether or not the collection is
    # new, so the structured-stats facet fields get indexed on an EXISTING
    # collection during a delta re-scrape too.
    indexes = [
        ("category",    PayloadSchemaType.KEYWORD),
        ("chunk_type",  PayloadSchemaType.KEYWORD),
        ("doc_type",    PayloadSchemaType.KEYWORD),
        ("entities",    PayloadSchemaType.KEYWORD),
        ("breadcrumb",  PayloadSchemaType.KEYWORD),
        ("title",       PayloadSchemaType.KEYWORD),
        # ── facets (structured-stats overhaul) ──
        ("subject",     PayloadSchemaType.KEYWORD),
        ("weapon_type", PayloadSchemaType.KEYWORD),
        ("dlc",         PayloadSchemaType.BOOL),
        ("scaling_str", PayloadSchemaType.KEYWORD),
        ("scaling_dex", PayloadSchemaType.KEYWORD),
        ("scaling_int", PayloadSchemaType.KEYWORD),
        ("scaling_fai", PayloadSchemaType.KEYWORD),
        ("scaling_arc", PayloadSchemaType.KEYWORD),
        ("weak_to",     PayloadSchemaType.KEYWORD),
        ("weight",      PayloadSchemaType.FLOAT),
        ("fp_cost",     PayloadSchemaType.INTEGER),
    ]
    for field, schema in indexes:
        try:
            client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
        except Exception:
            pass  # index already exists
    console.print(f"[dim]Payload indexes ensured ({len(indexes)}).[/dim]")

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
            # Retry transient network/timeout errors (common when seeding a remote
            # cluster over a home link) so one blip doesn't abort a long upload.
            for attempt in range(1, UPSERT_RETRIES + 1):
                try:
                    client.upsert(COLLECTION_NAME, points=points, wait=True)
                    break
                except Exception as e:
                    if attempt == UPSERT_RETRIES:
                        raise
                    backoff = min(2 ** attempt, 30)
                    console.print(
                        f"[yellow]batch {start}-{end} failed "
                        f"(attempt {attempt}/{UPSERT_RETRIES}): {type(e).__name__} — "
                        f"retrying in {backoff}s[/yellow]"
                    )
                    time.sleep(backoff)
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
    parser.add_argument("--embeddings-dir", type=str, default=None,
                        help="artifacts dir to upload (default data/embeddings); point at a "
                             "supplement dir to upsert only new points")
    args = parser.parse_args()
    run(recreate=args.recreate, embeddings_dir=args.embeddings_dir)


if __name__ == "__main__":
    main()