"""
backend/core/build_gazetteer.py

Builds the canonical entity gazetteer the resolver uses, from the entities[]
fields already present in chunks.jsonl. Run once (and re-run whenever the corpus
changes). The output is a small JSON the resolver loads at startup.

Why frequencies: when a fuzzy match is ambiguous between near-duplicate names
("Malenia" vs "Malenia, Blade of Miquella"), the resolver can prefer the variant
that actually appears most often as a filterable entity in the corpus — i.e. the
one most likely to return results from Qdrant.

Output:
    backend/core/data/gazetteer.json
        { "entities": { "<canonical name>": <count>, ... }, "count": N }

Usage:
    uv run python backend/core/build_gazetteer.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rich.console import Console

console = Console()

# chunks.jsonl lives in the data pipeline output
CHUNKS_FILE = Path("data_pipeline/data/chunks.jsonl")
OUT_DIR     = Path("backend/core/data")
OUT_FILE    = OUT_DIR / "gazetteer.json"


def build() -> None:
    if not CHUNKS_FILE.exists():
        console.print(f"[red]No chunks file at {CHUNKS_FILE}[/red]")
        return

    counter: Counter[str] = Counter()
    n_chunks = 0
    for line in CHUNKS_FILE.open(encoding="utf-8"):
        n_chunks += 1
        chunk = json.loads(line)
        for ent in chunk.get("entities", []):
            ent = ent.strip()
            if ent:
                counter[ent] += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "entities": dict(counter),     # name → frequency across all chunks
        "count": len(counter),
    }
    OUT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    console.print(f"[green]Built gazetteer[/green] from {n_chunks} chunks")
    console.print(f"  unique entities: {len(counter)}")
    console.print(f"  most common: {counter.most_common(8)}")
    console.print(f"  → {OUT_FILE}")


if __name__ == "__main__":
    build()