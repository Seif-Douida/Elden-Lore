"""
data_pipeline/patch_dlc.py

Recompute the `dlc` facet from the current _is_dlc heuristic and bake it into the
embeddings payloads FILE — then re-ingest with `upload.py --recreate`.

WHY NOT a live set_payload: a bulk `client.set_payload` over the whole collection
corrupted the Qdrant v1.16.1 HNSW index (vector search then 500'd with
"length must be greater than zero"); recovery required a full --recreate. So we
NEVER live-patch the collection. Instead we edit embeddings/payloads.jsonl (no
re-embed — the vectors are unchanged) and rebuild the collection cleanly.

Usage (from data_pipeline/):
    uv run python patch_dlc.py            # rewrite payloads.jsonl with fresh dlc
    uv run python patch_dlc.py --dry-run  # report counts only
    # then:
    uv run python upload.py --recreate    # rebuild the collection from the patched file
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chunker import _is_dlc

DATA_DIR = Path(__file__).parent / "data"
PAGES_FILE = DATA_DIR / "pages.jsonl"
PAYLOADS_FILE = DATA_DIR / "embeddings" / "payloads.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake fresh dlc into payloads.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pages = [json.loads(l) for l in PAGES_FILE.open(encoding="utf-8")]
    dlc_by_url = {p["url"]: _is_dlc(p) for p in pages if p.get("url")}
    print(f"pages: {len(pages)}  DLC pages: {sum(dlc_by_url.values())}")

    if not PAYLOADS_FILE.exists():
        print(f"[error] no payloads at {PAYLOADS_FILE} (run embed.py first)")
        return

    payloads = [json.loads(l) for l in PAYLOADS_FILE.open(encoding="utf-8")]
    changed = 0
    for p in payloads:
        new = dlc_by_url.get(p.get("url"), False)
        if p.get("dlc") != new:
            changed += 1
        p["dlc"] = new
    n_true = sum(1 for p in payloads if p.get("dlc"))
    print(f"payloads: {len(payloads)}  dlc=True {n_true}  (changed {changed})")

    if args.dry_run:
        print("[dry-run] payloads.jsonl not written.")
        return

    with PAYLOADS_FILE.open("w", encoding="utf-8") as f:
        for p in payloads:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("payloads.jsonl rewritten. Now run:  uv run python upload.py --recreate")


if __name__ == "__main__":
    main()
