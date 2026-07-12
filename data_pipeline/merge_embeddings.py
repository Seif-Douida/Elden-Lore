"""One-off: build a full embeddings artifact for the CURRENT chunks.jsonl by
REUSING unchanged vectors (keyed by content hash, since chunk_id is position-based
and shifts when a page gains/loses chunks) and taking freshly-embedded vectors for
the content-changed chunks. Output is a clean, complete set for `upload.py --recreate`.

    old (data/embeddings)          -> reused vectors, keyed by md5(url+raw_text)
    changed (data/embeddings_changed) -> new vectors for content that changed
    merged (data/embeddings_v2)    -> one vector+payload per current chunk

Every merged row is keyed to the CURRENT chunk's id + fresh to_payload(); the vector
is looked up by content hash so it always matches the text. Fails loudly if any
current chunk lacks a vector.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from embed import point_id, to_payload

DATA = Path("data")
OLD = DATA / "embeddings"
CHANGED = DATA / "embeddings_changed"
OUT = DATA / "embeddings_v2"
CHUNKS = DATA / "chunks.jsonl"


def chash(url: str, raw_text: str) -> str:
    return hashlib.md5((url + "\n" + raw_text).encode()).hexdigest()


def load_content_map(d: Path) -> dict[str, np.ndarray]:
    vecs = np.load(d / "vectors.npy")
    payloads = [json.loads(l) for l in (d / "payloads.jsonl").open(encoding="utf-8")]
    assert len(vecs) == len(payloads), f"{d}: vec/payload length mismatch"
    m: dict[str, np.ndarray] = {}
    for v, p in zip(vecs, payloads):
        m[chash(p["url"], p["raw_text"])] = v
    return m


def main() -> None:
    print("loading old + changed vector maps by content hash…")
    old_map = load_content_map(OLD)
    changed_map = load_content_map(CHANGED)
    print(f"  old_map={len(old_map):,}  changed_map={len(changed_map):,}")

    chunks = [json.loads(l) for l in CHUNKS.open(encoding="utf-8")]
    print(f"  current chunks={len(chunks):,}")

    out_vecs: list[np.ndarray] = []
    out_ids: list[int] = []
    out_payloads: list[dict] = []
    reused = new = missing = 0
    seen_ids: set[int] = set()

    for c in chunks:
        h = chash(c["url"], c["raw_text"])
        v = changed_map.get(h)
        if v is not None:
            new += 1
        else:
            v = old_map.get(h)
            if v is not None:
                reused += 1
        if v is None:
            missing += 1
            if missing <= 5:
                print(f"  MISSING vector for {c['title']} :: {c['raw_text'][:60]!r}")
            continue
        pid = point_id(c["chunk_id"])
        if pid in seen_ids:      # position-based ids should be unique per corpus
            print(f"  DUP id {pid} for {c['title']}")
        seen_ids.add(pid)
        out_vecs.append(v)
        out_ids.append(pid)
        out_payloads.append(to_payload(c))

    print(f"  reused={reused:,}  new={new:,}  missing={missing}")
    assert missing == 0, "some current chunks have no vector — aborting"
    assert len(out_ids) == len(chunks), "row count != chunk count"
    assert len(set(out_ids)) == len(out_ids), "duplicate point ids"

    vectors = np.asarray(out_vecs, dtype=np.float32)
    ids = np.asarray(out_ids, dtype=np.uint64)
    norms = np.linalg.norm(vectors, axis=1)
    assert not np.isnan(vectors).any() and not np.isinf(vectors).any(), "nan/inf in vectors"
    assert norms.min() > 0.99 and norms.max() < 1.01, f"bad norms: {norms.min()}..{norms.max()}"

    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "vectors.npy", vectors)
    np.save(OUT / "ids.npy", ids)
    with (OUT / "payloads.jsonl").open("w", encoding="utf-8") as f:
        for p in out_payloads:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nDone -> {OUT}")
    print(f"  vectors {vectors.shape}  ids {ids.shape}  payloads {len(out_payloads):,}")
    print(f"  norms {norms.min():.4f}..{norms.max():.4f}")


if __name__ == "__main__":
    main()
