"""
data_pipeline/embed.py

Stage 4a of the pipeline: EMBED (decoupled from upload).

Reads chunks.jsonl, embeds every chunk's `text` with bge-base-en-v1.5 on the
GPU, and writes the vectors + aligned ids + payloads to disk as a durable,
reusable artifact. This is the slow part — done once. If Qdrant ever needs
rebuilding, upload.py re-loads these files in seconds without re-embedding.

Outputs (all aligned by row order):
    data/embeddings/vectors.npy      float32 [N, 768]
    data/embeddings/ids.npy          uint64  [N]        (deterministic point ids)
    data/embeddings/payloads.jsonl   N lines            (one payload dict per row)
    data/embeddings/meta.json        run metadata + sanity report

Usage:
    uv run python data_pipeline/embed.py
    uv run python data_pipeline/embed.py --limit 100        # test slice
    uv run python data_pipeline/embed.py --batch 32         # smaller VRAM
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from common import DATA_DIR

console = Console()

# ── Configuration ─────────────────────────────────────────────────────────────

CHUNKS_FILE = DATA_DIR / "chunks.jsonl"
EMBED_DIR   = DATA_DIR / "embeddings"
VECTORS_FILE  = EMBED_DIR / "vectors.npy"
IDS_FILE      = EMBED_DIR / "ids.npy"
PAYLOADS_FILE = EMBED_DIR / "payloads.jsonl"
META_FILE     = EMBED_DIR / "meta.json"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
VECTOR_DIM  = 768
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "64"))


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


def point_id(chunk_id: str) -> int:
    """16-char hex chunk_id → uint64 point id (deterministic, idempotent)."""
    return int(chunk_id, 16)


# ── Payload projection ────────────────────────────────────────────────────────

def to_payload(chunk: dict) -> dict:
    return {
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


# ── Main ──────────────────────────────────────────────────────────────────────

def run(limit: Optional[int] = None) -> None:
    if not CHUNKS_FILE.exists():
        console.print(f"[red]No chunks file at {CHUNKS_FILE}. Run chunker.py first.[/red]")
        return

    chunks = [json.loads(l) for l in CHUNKS_FILE.open(encoding="utf-8")]
    if limit:
        chunks = chunks[:limit]
    n = len(chunks)
    console.print(f"[bold cyan]Embedding {n} chunks[/bold cyan]")

    from sentence_transformers import SentenceTransformer

    device = _select_device()
    console.print(f"[dim]Loading {EMBED_MODEL} on [bold]{device}[/bold] · batch={EMBED_BATCH}[/dim]")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    if device == "cuda":
        try:
            import torch
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            console.print(f"[dim]GPU: {gpu} ({vram:.1f} GB)[/dim]")
        except Exception:
            pass

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-allocate the vectors array; fill row by row.
    vectors = np.zeros((n, VECTOR_DIM), dtype=np.float32)
    ids = np.zeros(n, dtype=np.uint64)

    with PAYLOADS_FILE.open("w", encoding="utf-8") as pf, Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("Embedding…", total=n)
        for start in range(0, n, EMBED_BATCH):
            batch = chunks[start:start + EMBED_BATCH]
            texts = [c["text"] for c in batch]
            vecs = model.encode(
                texts, batch_size=EMBED_BATCH,
                normalize_embeddings=True, show_progress_bar=False,
            )
            for i, (chunk, vec) in enumerate(zip(batch, vecs)):
                row = start + i
                vectors[row] = vec
                ids[row] = point_id(chunk["chunk_id"])
                pf.write(json.dumps(to_payload(chunk), ensure_ascii=False) + "\n")
            progress.update(task, advance=len(batch))

    # ── Sanity checks ────────────────────────────────────────────────────────
    console.print("\n[bold]Sanity checks:[/bold]")
    n_nan = int(np.isnan(vectors).any(axis=1).sum())
    n_inf = int(np.isinf(vectors).any(axis=1).sum())
    norms = np.linalg.norm(vectors, axis=1)
    n_zero = int((norms < 1e-6).sum())
    unique_ids = len(np.unique(ids))

    checks = {
        "rows": n,
        "vector_dim": int(vectors.shape[1]),
        "rows_with_nan": n_nan,
        "rows_with_inf": n_inf,
        "zero_norm_rows": n_zero,
        "unique_ids": unique_ids,
        "id_collisions": n - unique_ids,
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }
    ok = (n_nan == 0 and n_inf == 0 and n_zero == 0 and checks["id_collisions"] == 0
          and vectors.shape[1] == VECTOR_DIM)
    for key, val in checks.items():
        console.print(f"  {key}: {val}")
    console.print(f"  [bold]{'PASS' if ok else 'FAIL'}[/bold]")

    # Normalized vectors should have norm ≈ 1.0
    if abs(checks["norm_min"] - 1.0) > 0.01 or abs(checks["norm_max"] - 1.0) > 0.01:
        console.print("  [yellow]warning: norms not ≈1.0 — check normalize_embeddings[/yellow]")

    # ── Save artifacts ───────────────────────────────────────────────────────
    np.save(VECTORS_FILE, vectors)
    np.save(IDS_FILE, ids)
    meta = {
        "model": EMBED_MODEL,
        "vector_dim": VECTOR_DIM,
        "count": n,
        "device": device,
        "checks": checks,
        "ok": ok,
    }
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    console.print(f"\n[bold green]Done![/bold green] Saved to {EMBED_DIR}/")
    console.print(f"  vectors.npy   {vectors.shape} float32 (~{vectors.nbytes/1e6:.0f} MB)")
    console.print(f"  ids.npy       {ids.shape} uint64")
    console.print(f"  payloads.jsonl {n} lines")
    if not ok:
        console.print("[red]Sanity checks FAILED — do not upload; investigate first.[/red]")
    else:
        console.print("[green]Sanity checks passed — ready for upload.py[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Elden Ring — embed chunks to disk")
    parser.add_argument("--limit", type=int, default=None, help="embed only first N chunks")
    parser.add_argument("--batch", type=int, default=None, help="override embed batch size")
    args = parser.parse_args()
    if args.batch:
        global EMBED_BATCH
        EMBED_BATCH = args.batch
    run(limit=args.limit)


if __name__ == "__main__":
    main()