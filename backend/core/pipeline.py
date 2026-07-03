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

from dataclasses import dataclass, field
from typing import Optional

from core.router import Router, ResolvedDecision
from core.retriever import Retriever, RetrievedChunk

DEFAULT_TOP_K = 8


@dataclass
class PipelineResult:
    question:       str
    decision:       ResolvedDecision
    chunks:         list[RetrievedChunk]
    retrieved:      bool                       # did we hit the retriever at all?
    entity_fallback: bool = False              # did the entity filter fall back?
    notes:          list[str] = field(default_factory=list)


class Pipeline:
    def __init__(self) -> None:
        self._router = Router()
        self._retriever = Retriever()

    def run(self, question: str, history: Optional[str] = None,
            k: int = DEFAULT_TOP_K) -> PipelineResult:
        decision = self._router.route(question, history)
        notes: list[str] = []

        # Greetings / chitchat — no retrieval.
        if not decision.needs_retrieval:
            notes.append("needs_retrieval=False → skipped retrieval")
            return PipelineResult(question, decision, [], retrieved=False, notes=notes)

        has_intent = decision.intent is not None
        boost_kwargs = dict(
            intent=decision.intent,
            boost_section=has_intent,
            boost_breadcrumb=has_intent,
            boost_chunktype=has_intent,
            diversity_penalty=True,            # always helpful, intent or not
        )

        common = dict(
            k=k,
            category=decision.category_hint,
            chunk_type=decision.chunk_type_bias,
            **boost_kwargs,
        )

        # Use a focused retrieval query when the router stripped context noise
        # (e.g. "I am in X, how do I reach Y?" → embed only "how to reach Y").
        # Falls back to the raw question when retrieval_query is None.
        embed_query = decision.retrieval_query or question

        # ── Attempt 1: hard entity filter (if any entity resolved) ───────────
        entity_fallback = False
        chunks: list[RetrievedChunk] = []
        if decision.entity_focus:
            chunks = self._retriever.retrieve(
                embed_query, entities=decision.entity_focus, **common
            )
            if not chunks:
                # ── Fallback: drop the entity filter, keep intent/boosts ─────
                entity_fallback = True
                notes.append(
                    f"entity filter {decision.entity_focus} returned empty → "
                    "fell back to no-entity search (check entity casing/coverage)"
                )
                chunks = self._retriever.retrieve(embed_query, **common)
        else:
            chunks = self._retriever.retrieve(embed_query, **common)

        return PipelineResult(
            question=question,
            decision=decision,
            chunks=chunks,
            retrieved=True,
            entity_fallback=entity_fallback,
            notes=notes,
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