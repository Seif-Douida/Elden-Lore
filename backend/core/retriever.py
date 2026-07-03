"""
backend/core/retriever.py

Thin retriever over the hand-built 'elden_ring' Qdrant collection.

Why not LangChain's QdrantVectorStore: our collection uses a FLAT payload
(url, title, entities, image_url ... all top-level), while that wrapper expects
page_content + nested metadata. So we read Qdrant directly with qdrant-client,
keep full control over the bge query-side instruction (the correctness-critical
bit), and return native LangChain `Document` objects so everything downstream —
prompts, chains, the agent, the LLM layer — is pure LangChain.

Retrieval design (after experimentation — see notes):
  1. dense vector search (bge-base-en-v1.5) → semantic recall
  2. payload filters on entities / category / chunk_type / doc_type → exact match
     (this is the real engine: entity-filtering beat a neural cross-encoder
      reranker outright on entity & dialogue queries)
  3. STRUCTURED BOOST (optional) → light, additive re-ranking *within* the
     filtered set using our own clean metadata (section_heading, breadcrumb,
     chunk_type) plus a page-diversity penalty. Driven by an explicit `intent`.

The neural cross-encoder reranker was tested and DROPPED: it over-weighted
vocabulary overlap with the question (e.g. boosting Sorceress Sellen's
"questline" chunk for a Ranni question) and hurt precision. The structured
boost is entity-aware by construction and uses small additive weights, so it
reorders near-ties without catapulting wrong-entity chunks.

Run the harness:
    uv run python backend/core/retriever.py --compare --intent drops \
        --entity "Starscourge Radahn" "what does Radahn drop"
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
COLLECTION  = os.getenv("QDRANT_COLLECTION_NAME", "elden_ring")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

# bge-*-v1.5 short-query→passage retrieval: prepend this to the QUERY only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"

DEFAULT_TOP_K = 8
# When boosting, fetch a wider candidate pool so reordering has material.
DEFAULT_FETCH_K = 40

# ── Boost weights (small + additive; vector score 0.6–0.78 stays dominant) ────
SECTION_BOOST     = 0.05   # chunk's section_heading matches the query intent
BREADCRUMB_BOOST  = 0.03   # chunk's breadcrumb/category matches the query intent
CHUNKTYPE_BOOST   = 0.05   # chunk_type matches the intent (e.g. dialogue)
DIVERSITY_PENALTY = 0.04   # per extra chunk from the same page beyond the 2nd

# Intent profiles: which structural values each query-intent should favour.
# Matching is case-insensitive substring against the chunk's fields.
INTENT_PROFILES: dict[str, dict[str, set[str]]] = {
    "drops": {
        "sections":    {"drops", "combat information", "overview"},
        "crumb":       {"bosses", "items"},
        "chunk_types": set(),
    },
    "strategy": {
        "sections":    {"fight strategy", "attacks & counters", "attacks", "combat", "strategy"},
        "crumb":       {"bosses"},
        "chunk_types": set(),
    },
    "location": {
        "sections":    {"location", "where to find", "overview"},
        "crumb":       {"locations"},
        "chunk_types": set(),
    },
    "dialogue": {
        "sections":    {"dialogue"},
        "crumb":       {"npcs"},
        "chunk_types": {"dialogue"},
    },
    "lore": {
        "sections":    {"lore", "notes & trivia", "notes", "trivia", "story", "overview"},
        "crumb":       {"lore"},
        "chunk_types": set(),
    },
    "quest": {
        "sections":    {"quest", "questline", "walkthrough", "side quests"},
        "crumb":       {"npcs", "walkthrough"},
        "chunk_types": set(),
    },
    "summary": {
        "sections":    {"overview", "list", "all bosses", "main bosses",
                        "achievement", "complete list", "full list", "all"},
        "crumb":       set(),
        "chunk_types": {"body", "item_desc"},
    },
}
VALID_INTENTS = sorted(INTENT_PROFILES.keys())


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


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    score:           float            # bi-encoder (vector) similarity
    boost:           float            # additive structural boost (0.0 if none)
    final_score:     float            # score + boost (the ranking key)
    text:            str
    raw_text:        str
    url:             str
    title:           str
    category:        str
    doc_type:        str
    breadcrumb:      list[str]
    section_heading: str
    chunk_type:      str
    entities:        list[str]
    image_url:       Optional[str]
    source_type:     str


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """Bi-encoder search + payload filters + optional structured boost."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient

        self._device = _select_device()
        self._model = SentenceTransformer(EMBED_MODEL, device=self._device)
        self._client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            https=False,
            check_compatibility=False,
        )

    # ── query embedding ──────────────────────────────────────────────────────
    def embed_query(self, query: str):
        text = f"{QUERY_INSTRUCTION} {query}"
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    # ── filter builder ───────────────────────────────────────────────────────
    def _build_filter(self, entities, category, chunk_type, doc_type):
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
        must = []
        if entities:
            must.append(FieldCondition(key="entities", match=MatchAny(any=entities)))
        if category:
            must.append(FieldCondition(key="category", match=MatchValue(value=category)))
        if chunk_type:
            must.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))
        if doc_type:
            must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
        return Filter(must=must) if must else None

    # ── structural boost ─────────────────────────────────────────────────────
    @staticmethod
    def _apply_boost(
        candidates: list[RetrievedChunk],
        intent: Optional[str],
        boost_section: bool,
        boost_breadcrumb: bool,
        boost_chunktype: bool,
        diversity_penalty: bool,
    ) -> None:
        """Mutates candidates: sets .boost and .final_score, then they get sorted."""
        profile = INTENT_PROFILES.get(intent) if intent else None

        # Per-signal additive boosts driven by the intent profile.
        for c in candidates:
            b = 0.0
            if profile:
                sec = c.section_heading.lower()
                crumb = " / ".join(c.breadcrumb).lower()
                if boost_section and any(s in sec for s in profile["sections"]):
                    b += SECTION_BOOST
                if boost_breadcrumb and any(k in crumb for k in profile["crumb"]):
                    b += BREADCRUMB_BOOST
                if boost_chunktype and c.chunk_type in profile["chunk_types"]:
                    b += CHUNKTYPE_BOOST
            c.boost = b
            c.final_score = c.score + b

        # Page-diversity penalty: demote the 3rd+ chunk from the same page.
        # Applied after the intent boosts, in current (vector) order.
        if diversity_penalty:
            seen: dict[str, int] = {}
            for c in candidates:
                n = seen.get(c.url, 0)
                if n >= 2:
                    c.boost -= DIVERSITY_PENALTY
                    c.final_score = c.score + c.boost
                seen[c.url] = n + 1

    # ── core retrieve ────────────────────────────────────────────────────────
    def retrieve(
        self,
        query:      str,
        k:          int = DEFAULT_TOP_K,
        entities:   Optional[list[str]] = None,
        category:   Optional[str] = None,
        chunk_type: Optional[str] = None,
        doc_type:   Optional[str] = None,
        intent:     Optional[str] = None,
        boost_section:     bool = False,
        boost_breadcrumb:  bool = False,
        boost_chunktype:   bool = False,
        diversity_penalty: bool = False,
        fetch_k:    int = DEFAULT_FETCH_K,
    ) -> list[RetrievedChunk]:
        vector = self.embed_query(query)
        qfilter = self._build_filter(entities, category, chunk_type, doc_type)

        any_boost = (
            (intent and (boost_section or boost_breadcrumb or boost_chunktype))
            or diversity_penalty
        )
        limit = fetch_k if any_boost else k

        result = self._client.query_points(
            collection_name=COLLECTION,
            query=vector,
            limit=limit,
            query_filter=qfilter,
            with_payload=True,
        )

        candidates: list[RetrievedChunk] = []
        for pt in result.points:
            p = pt.payload or {}
            candidates.append(RetrievedChunk(
                score=pt.score,
                boost=0.0,
                final_score=pt.score,
                text=p.get("text", ""),
                raw_text=p.get("raw_text", ""),
                url=p.get("url", ""),
                title=p.get("title", ""),
                category=p.get("category", ""),
                doc_type=p.get("doc_type", ""),
                breadcrumb=p.get("breadcrumb", []),
                section_heading=p.get("section_heading", ""),
                chunk_type=p.get("chunk_type", ""),
                entities=p.get("entities", []),
                image_url=p.get("image_url"),
                source_type=p.get("source_type", "wiki"),
            ))

        if not any_boost or not candidates:
            return candidates[:k]

        self._apply_boost(
            candidates, intent,
            boost_section, boost_breadcrumb, boost_chunktype, diversity_penalty,
        )
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates[:k]

    # ── LangChain bridge ─────────────────────────────────────────────────────
    def retrieve_documents(self, query: str, k: int = DEFAULT_TOP_K, **kwargs):
        from langchain_core.documents import Document
        out = []
        for c in self.retrieve(query, k=k, **kwargs):
            out.append(Document(
                page_content=c.raw_text,
                metadata={
                    "score": c.score,
                    "boost": c.boost,
                    "final_score": c.final_score,
                    "url": c.url,
                    "title": c.title,
                    "category": c.category,
                    "doc_type": c.doc_type,
                    "breadcrumb": c.breadcrumb,
                    "section_heading": c.section_heading,
                    "chunk_type": c.chunk_type,
                    "entities": c.entities,
                    "image_url": c.image_url,
                    "source_type": c.source_type,
                },
            ))
        return out


# ── CLI test harness ──────────────────────────────────────────────────────────

_SAMPLE_QUERIES = [
    "What does Starscourge Radahn drop when defeated?",
    "How do I progress Ranni the Witch's questline?",
    "Where can I find the Meteorite Staff?",
    "What is the lore of Caelid and the scarlet rot?",
    "What does Iron Fist Alexander say when you first meet him?",
]


def _format_results(query: str, chunks: list[RetrievedChunk]) -> str:
    lines = [f"Q: {query}"]
    if not chunks:
        lines += ["  (no results)", ""]
        return "\n".join(lines)
    for i, c in enumerate(chunks, 1):
        crumb = " / ".join(c.breadcrumb) if c.breadcrumb else "—"
        bs = f"  boost={c.boost:+.3f} final={c.final_score:.3f}" if c.boost else ""
        lines.append(f"  {i}. vec={c.score:.3f}{bs}  [{c.chunk_type}]  {c.title} · {c.section_heading}")
        lines.append(f"     path: {crumb}")
        lines.append(f"     url:  {c.url}")
        snippet = c.raw_text[:200].strip().replace("\n", " ")
        lines.append(f"     text: {snippet}…")
        if c.entities:
            lines.append(f"     entities: {', '.join(c.entities[:10])}")
        lines.append("")
    return "\n".join(lines)


def _format_compare(query, baseline, boosted) -> str:
    lines = [f"Q: {query}", "", "  BASELINE (filtered, no boost):"]
    for i, c in enumerate(baseline, 1):
        lines.append(f"    {i}. vec={c.score:.3f}  {c.title} · {c.section_heading}  [{c.chunk_type}]")
    lines += ["", "  BOOSTED (structural reorder):"]
    for i, c in enumerate(boosted, 1):
        lines.append(
            f"    {i}. final={c.final_score:.3f} (vec={c.score:.3f}, boost={c.boost:+.3f})  "
            f"{c.title} · {c.section_heading}  [{c.chunk_type}]"
        )
    lines += ["", "  Full boosted results:", _format_results(query, boosted), "=" * 70, ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Test the Elden Ring retriever")
    p.add_argument("query", nargs="*", help="Query text (omit to run the sample set)")
    p.add_argument("-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--fetch-k", type=int, default=DEFAULT_FETCH_K)
    p.add_argument("--entity", action="append", help="filter: entity (repeatable)")
    p.add_argument("--category", help="filter: lore|item|boss|quest|location")
    p.add_argument("--chunk-type", help="filter: body|dialogue|item_desc")
    p.add_argument("--doc-type", help="filter: page|walkthrough")
    p.add_argument("--intent", choices=VALID_INTENTS, help="explicit query intent for boosting")
    p.add_argument("--boost-section", action="store_true")
    p.add_argument("--boost-breadcrumb", action="store_true")
    p.add_argument("--boost-chunktype", action="store_true")
    p.add_argument("--diversity", action="store_true", help="page-diversity penalty")
    p.add_argument("--boost-all", action="store_true", help="enable all four boost signals")
    p.add_argument("--compare", action="store_true",
                   help="baseline (no boost) vs boosted, side by side")
    p.add_argument("--out", default="retriever_output.txt")
    args = p.parse_args()

    from rich.console import Console
    console = Console()
    console.print(f"[dim]Loading model, connecting to {QDRANT_HOST}:{QDRANT_PORT}…[/dim]")
    r = Retriever()
    console.print(f"[dim]device: {r._device}[/dim]")

    b_section = args.boost_section or args.boost_all
    b_crumb   = args.boost_breadcrumb or args.boost_all
    b_ctype   = args.boost_chunktype or args.boost_all
    b_div     = args.diversity or args.boost_all

    filters = dict(entities=args.entity, category=args.category,
                   chunk_type=args.chunk_type, doc_type=args.doc_type)
    boosts = dict(intent=args.intent, boost_section=b_section, boost_breadcrumb=b_crumb,
                  boost_chunktype=b_ctype, diversity_penalty=b_div, fetch_k=args.fetch_k)
    queries = [" ".join(args.query)] if args.query else _SAMPLE_QUERIES
    active = ", ".join(f"{k}={v}" for k, v in filters.items() if v) or "none"
    active_boosts = ", ".join(n for n, on in
                              [("section", b_section), ("breadcrumb", b_crumb),
                               ("chunktype", b_ctype), ("diversity", b_div)] if on) or "none"

    if args.compare:
        blocks = []
        for q in queries:
            baseline = r.retrieve(q, k=args.k, **filters)
            boosted  = r.retrieve(q, k=args.k, **filters, **boosts)
            blocks.append(_format_compare(q, baseline, boosted))
        header = [
            "Elden Ring retriever — BOOST COMPARISON",
            f"top_k: {args.k}  fetch_k: {args.fetch_k}  intent: {args.intent}  "
            f"boosts: {active_boosts}  filters: {active}",
            "=" * 70, "",
        ]
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n".join(blocks))
        console.print(f"[green]Wrote comparison for {len(queries)} query(ies) → {args.out}[/green]")
        return

    runs = [(q, r.retrieve(q, k=args.k, **filters, **boosts)) for q in queries]
    header = [
        "Elden Ring retriever — results",
        f"top_k: {args.k}  intent: {args.intent}  boosts: {active_boosts}  filters: {active}",
        "=" * 70, "",
    ]
    body = "\n".join(_format_results(q, chunks) for q, chunks in runs)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + body)
    console.print(f"[green]Wrote results for {len(runs)} query(ies) → {args.out}[/green]")


if __name__ == "__main__":
    main()