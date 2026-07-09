"""
backend/core/pipeline.py

Wires the router and the retriever into one "question in → chunks out" flow.
This is the integration seam between the two validated components.

Rules (decided from stress-test evidence):
  - entity_focus → HARD filter, with graceful FALLBACK: if the entity filter
    returns nothing (wrong/missing/mis-cased entity), retry WITHOUT the entity
    filter so the user still gets good vector+boost results instead of empty.
    Fallback is logged — repeated fallbacks signal an entity-resolution problem.
  - boosts → all four (section, breadcrumb, chunktype, diversity) whenever the
    router supplies an intent; diversity-only when it doesn't.
  - needs_retrieval == False → short-circuit, return no chunks (greetings).

Output bundles the chunks with the routing decision and diagnostics, ready for
the generation layer to build a prompt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from core.router import Router, ResolvedDecision
from core.retriever import Retriever, RetrievedChunk

DEFAULT_TOP_K = 8
# Broad questions (enumerations, questlines, lore overviews) need more context than a
# focused lookup. fetch_k=40 in the retriever, so returning more is free candidate-wise.
BROAD_TOP_K = 14
# summary (enumerations) and quest (multi-step guides) genuinely need breadth.
# lore is usually a SINGLE entity — extra chunks just surface generic hub pages
# (e.g. "Hornsent" → the generic "Lore" page outranked the Hornsent page at k=14).
BROAD_INTENTS = {"summary", "quest"}


def _k_for(intent: Optional[str], k: int) -> int:
    return max(k, BROAD_TOP_K) if intent in BROAD_INTENTS else k


@dataclass
class PipelineResult:
    question:       str
    decision:       ResolvedDecision
    chunks:         list[RetrievedChunk]
    retrieved:      bool                       # did we hit the retriever at all?
    entity_fallback: bool = False              # did the entity filter fall back?
    notes:          list[str] = field(default_factory=list)
    timings:        dict = field(default_factory=dict)   # {router_ms, retrieval_ms}
    roster:         list[str] = field(default_factory=list)  # enumeration titles (A)


class Pipeline:
    def __init__(self) -> None:
        self._router = Router()
        self._retriever = Retriever()

    def run(self, question: str, history: Optional[str] = None,
            k: int = DEFAULT_TOP_K) -> PipelineResult:
        _t0 = time.perf_counter()
        decision = self._router.route(question, history)
        router_ms = int((time.perf_counter() - _t0) * 1000)
        timings = {"router_ms": router_ms, "retrieval_ms": 0}
        notes: list[str] = []

        # Greetings / chitchat — no retrieval.
        if not decision.needs_retrieval:
            notes.append("needs_retrieval=False → skipped retrieval")
            return PipelineResult(question, decision, [], retrieved=False,
                                  notes=notes, timings=timings)

        # Broad intents (summary/quest/lore) get more chunks for completeness.
        k = _k_for(decision.intent, k)

        has_intent = decision.intent is not None
        boost_kwargs = dict(
            intent=decision.intent,
            boost_section=has_intent,
            boost_breadcrumb=has_intent,
            boost_chunktype=has_intent,
            diversity_penalty=True,            # always helpful, intent or not
        )

        # category_hint is a HARD Qdrant filter, and the router's category guess is
        # unreliable — e.g. "Who is Vyke?" → category_hint=lore, but the Vyke NPC page
        # is category 'quest', so an entity+category AND-filter excluded the very page
        # we wanted. When an entity is resolved it's already the precision signal, so
        # DON'T also hard-filter by category (which can only over-constrain). Also drop
        # it for BROAD intents: "which bosses are mandatory" gets category_hint='boss',
        # but the answer lives in the Bosses HUB page's "Mandatory Bosses" section (a
        # different category), so category=boss would exclude the very chunk we need.
        # Keep the category filter only for entity-less, narrow-intent queries.
        use_category = not decision.entity_focus and decision.intent not in BROAD_INTENTS

        # chunk_type_bias is ALSO a HARD filter, and Gemma sometimes mis-sets it — e.g.
        # "Why was Godwyn killed?" got chunk_type_bias='item_desc'. That AND-filtered the
        # Godwyn entity down to ~nothing, the fallback dropped the entity but KEPT
        # chunk_type='item_desc', so retrieval scanned every item description and returned
        # junk ("Skills · Item Description") → the model refused garbage context. Only
        # 'dialogue' is a reliable hard constraint (dialogue chunks are sparse and
        # distinctly wanted for "what does X say"); for body/item_desc the intent boost
        # already handles preference, so don't let a stray bias become a hard filter.
        chunk_type_filter = decision.chunk_type_bias if decision.chunk_type_bias == "dialogue" else None
        common = dict(
            k=k,
            category=decision.category_hint if use_category else None,
            chunk_type=chunk_type_filter,
            **boost_kwargs,
        )

        # Use a focused retrieval query when the router stripped context noise
        # (e.g. "I am in X, how do I reach Y?" → embed only "how to reach Y").
        # Falls back to the raw question when retrieval_query is None.
        embed_query = decision.retrieval_query or question

        # For enumeration, the roster is authoritative — don't let a mis-resolved
        # CATEGORY entity (e.g. "katanas" → "Katar") narrow the descriptive
        # retrieval; search broadly so the supporting context is on-topic.
        entities = [] if decision.enumerate_group else decision.entity_focus

        # ── Attempt 1: hard entity filter (if any entity resolved) ───────────
        entity_fallback = False
        chunks: list[RetrievedChunk] = []
        _tr = time.perf_counter()
        if entities:
            chunks = self._retriever.retrieve(
                embed_query, entities=entities, **common
            )
            if not chunks:
                # ── Fallback: drop the entity filter, keep intent/boosts ─────
                entity_fallback = True
                notes.append(
                    f"entity filter {entities} returned empty → "
                    "fell back to no-entity search (check entity casing/coverage)"
                )
                chunks = self._retriever.retrieve(embed_query, **common)
        else:
            chunks = self._retriever.retrieve(embed_query, **common)
        timings["retrieval_ms"] = int((time.perf_counter() - _tr) * 1000)

        # Enumeration: for "how many / list all X", get the precise roster from
        # metadata (the vector chunks above still provide descriptions). Empty →
        # ignore and fall back to the normal answer.
        roster: list[str] = []
        if decision.enumerate_group:
            roster = self._retriever.enumerate_titles(
                decision.enumerate_group, category=decision.category_hint
            )
            if not roster:  # category_hint may be wrong/too strict — retry without it
                roster = self._retriever.enumerate_titles(decision.enumerate_group)
            if roster:
                notes.append(f"enumerate '{decision.enumerate_group}' → {len(roster)} titles")

        return PipelineResult(
            question=question,
            decision=decision,
            chunks=chunks,
            retrieved=True,
            entity_fallback=entity_fallback,
            notes=notes,
            timings=timings,
            roster=roster,
        )


# ── CLI test harness ──────────────────────────────────────────────────────────

_SAMPLES = [
    "how do I beat Melania",
    "what does Radahn drop",
    "where can I find the meteorite staff",
    "what does Alexander say when you meet him",
    "tell me the story of Queen Marika",
    "hello there",
]


def _format(res: PipelineResult) -> str:
    d = res.decision
    lines = [
        f"Q: {res.question}",
        f"  routed: entities={d.entity_focus} intent={d.intent} "
        f"chunk_type={d.chunk_type_bias} category={d.category_hint} "
        f"tone={d.tone} needs_image={d.needs_image}",
        f"  raw_mentions={d.raw_mentions} used_call2={d.used_call2}",
    ]
    if res.notes:
        for n in res.notes:
            lines.append(f"  note: {n}")
    if not res.retrieved:
        lines.append("  (no retrieval)")
        lines.append("")
        return "\n".join(lines)
    lines.append(f"  fallback={res.entity_fallback}  chunks={len(res.chunks)}")
    for i, c in enumerate(res.chunks, 1):
        bs = f" final={c.final_score:.3f}" if c.boost else ""
        lines.append(f"    {i}. vec={c.score:.3f}{bs}  [{c.chunk_type}]  "
                     f"{c.title} · {c.section_heading}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Test the router→retriever pipeline")
    p.add_argument("question", nargs="*", help="Question (omit for sample set)")
    p.add_argument("-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="pipeline_output.txt")
    args = p.parse_args()

    from rich.console import Console
    console = Console()
    console.print("[dim]Loading pipeline (router + retriever)…[/dim]")
    pipe = Pipeline()

    questions = [" ".join(args.question)] if args.question else _SAMPLES
    blocks = []
    for q in questions:
        res = pipe.run(q, k=args.k)
        blocks.append(_format(res))
        fb = " [yellow](fallback)[/yellow]" if res.entity_fallback else ""
        console.print(f"[green]✓[/green] {q}  [dim]chunks={len(res.chunks)}[/dim]{fb}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("Router → Retriever pipeline\n" + "=" * 70 + "\n\n" + "\n".join(blocks))
    console.print(f"\n[green]Wrote {len(questions)} results → {args.out}[/green]")


if __name__ == "__main__":
    main()