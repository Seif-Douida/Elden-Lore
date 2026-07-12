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

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from core.router import Router, ResolvedDecision
from core.retriever import Retriever, RetrievedChunk

# Facet whitelist — the LLM sometimes emits junk keys ('enumerate_group' inside
# facets) or malformed ones; only these reach the retriever as hard filters.
_FACET_KEYS = {"dlc", "weapon_type", "weak_to", "strong_vs",
               "weight_min", "weight_max", "fp_cost_min", "fp_cost_max"}
_SCALING_KEY = re.compile(r"^scaling_(str|dex|int|fai|arc)_(min|max)$")
_SORT_VALUES = {"weight_desc", "weight_asc", "fp_cost_desc", "fp_cost_asc"}


def _clean_facets(facets) -> Optional[dict]:
    """Keep only well-formed facet keys the retriever understands. Returns None when
    nothing valid remains (so a hallucinated facet can't empty the retrieval)."""
    if not isinstance(facets, dict):
        return None
    out: dict = {}
    for k, v in facets.items():
        if k == "sort":
            if isinstance(v, str) and v in _SORT_VALUES:
                out[k] = v
        elif k in _FACET_KEYS or _SCALING_KEY.match(k):
            out[k] = v
    return out or None


# Deterministic superlative detection — the Gemma router intermittently omits the
# sort/group for "heaviest/lightest/cheapest X", and then improvises a WRONG ranking
# (e.g. ranking armor by Robustness, not weight). Detect the pattern in code so it's
# reliable regardless of the LLM.
_SUPERLATIVE_RE = re.compile(r"\b(heaviest|lightest|cheapest)\b", re.I)
_SUPERLATIVE_CAT = [
    (re.compile(r"\bsorcer", re.I),      "Sorceries"),
    (re.compile(r"\bincantation", re.I), "Incantations"),
    (re.compile(r"\barmou?r|\bhelm|\bgauntlet|\bgreaves|\bchest\b|\bleg armou?r", re.I), "Armor"),
    (re.compile(r"\bweapon|\bsword|\bkatana|\bgreatsword|\baxe|\bspear|\bhalberd|"
                r"\bdagger|\bhammer|\bbow|\bstaff|\bseal|\bfist|\bwhip|\breaper|"
                r"\bflail|\btwinblade", re.I), "Weapons"),
]


def _detect_superlative(question: str) -> tuple[Optional[str], Optional[str]]:
    """('weight_desc', 'Armor') for 'heaviest armor', etc. → (sort, group)."""
    m = _SUPERLATIVE_RE.search(question or "")
    if not m:
        return None, None
    word = m.group(1).lower()
    sort = {"heaviest": "weight_desc", "lightest": "weight_asc",
            "cheapest": "fp_cost_asc"}[word]
    group = next((g for pat, g in _SUPERLATIVE_CAT if pat.search(question)), None)
    return sort, group


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

        # Relational facets: the LLM sometimes emits junk keys (e.g. put
        # 'enumerate_group' INSIDE facets) or a valid key with a garbage value
        # ('weapon_type': 'Dexterity scaling weapons'), which as a HARD filter matches
        # nothing → empty context → "couldn't find anything". Whitelist the keys, then
        # split off the superlative `sort`.
        facets = _clean_facets(decision.facets)
        sort = facets.pop("sort", None) if facets else None
        facet_filter = facets or None

        # Deterministic superlative override — the router intermittently omits the
        # sort/group for "heaviest/lightest/cheapest X" and then improvises a wrong
        # ranking (armor by Robustness, not weight). Fill in from the question text
        # when the LLM didn't, so the pattern is reliable regardless of Gemma.
        det_sort, det_group = _detect_superlative(question)
        if det_sort and not sort:
            sort = det_sort
            notes.append(f"superlative detected → sort={det_sort} group={det_group}")

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

        # enumerate_group is meant to be a CATEGORY ("Katanas"), but the router
        # sometimes echoes the resolved item itself ("how many somber smithing stones"
        # → enumerate_group='Somber Ancient Dragon Smithing Stone' AND entity=[same]).
        # Enumerating a single item's own title yields nothing useful and — worse —
        # would drop the entity below → broad search → messy walkthrough chunks. When
        # enumerate_group just names the resolved entity, it's a lookup, not a roster.
        enum_group = decision.enumerate_group
        if enum_group and decision.entity_focus and any(
            enum_group.strip().lower() == e.strip().lower() for e in decision.entity_focus
        ):
            enum_group = None
            notes.append("enumerate_group == entity → treated as lookup, not enumeration")

        # Drop the entity filter ONLY for real enumeration (where a mis-resolved
        # CATEGORY entity like "katanas"→"Katar" would wrongly narrow). For a facet
        # query with a SPECIFIC resolved entity ("stats of the Moonveil" + a spurious
        # weapon_type facet), keep the entity — a named entity beats a hallucinated facet.
        entities = [] if enum_group else list(decision.entity_focus)

        # "I am in X, how do I reach Y?" — the router often resolves the ORIGIN (X)
        # as the entity and hard-filters retrieval to it, burying the destination Y.
        # The rewritten query frames X as an origin ("… from X"), so drop any entity
        # that appears only as a "from <entity>" origin — let the query drive to Y.
        rq_low = (decision.retrieval_query or "").lower()
        if entities and "from " in rq_low:
            kept = [e for e in entities if f"from {e.lower()}" not in rq_low
                    and f"from the {e.lower()}" not in rq_low]
            if kept != entities:
                notes.append(f"dropped origin entity from filter (was {entities}, now {kept})")
                entities = kept

        # ── Retrieval with graceful fallback ─────────────────────────────────
        entity_fallback = False
        _tr = time.perf_counter()
        if entities:
            # A specific named entity IS the precision signal — retrieve its own
            # chunks and do NOT also hard-filter by facet. A stray facet ('how many
            # somber stones in the DLC' → dlc=True) would exclude the entity's own
            # page (the smithing-stone item's DLC-locations chunk isn't dlc-tagged)
            # and surface tangential DLC walkthroughs instead. The facet still drives
            # the ROSTER below; here the entity wins.
            chunks = self._retriever.retrieve(embed_query, entities=entities, **common)
            if not chunks:
                entity_fallback = True
                notes.append(f"entity filter {entities} empty → no-entity search")
                chunks = self._retriever.retrieve(embed_query, facets=facet_filter, **common)
        else:
            chunks = self._retriever.retrieve(embed_query, facets=facet_filter, **common)
            # A facet that matched nothing (bad LLM value) shouldn't leave the answer
            # context-less — the roster is authoritative for the LIST anyway.
            if not chunks and facet_filter:
                notes.append(f"facet {facet_filter} empty → no-facet search")
                chunks = self._retriever.retrieve(embed_query, **common)
        timings["retrieval_ms"] = int((time.perf_counter() - _tr) * 1000)

        # Enumeration / relational / superlative: build a precise roster from
        # metadata (the vector chunks above still provide descriptions). Empty →
        # ignore and fall back to the normal answer.
        roster: list[str] = []
        # The deterministic detector's category ('Armor', 'Weapons', 'Sorceries')
        # maps to a REAL breadcrumb; the router's free-text enumerate_group is often
        # bogus ('Armor Sets', 'Dexterity scaling weapons') and matches nothing. So
        # for a superlative, trust the detector; otherwise use enumerate_group.
        group = det_group or enum_group
        if sort:
            # Superlative: "heaviest armor" → sort='weight_desc'. Rank by the facet.
            facet_name, direction = sort.rsplit("_", 1)
            ranked = self._retriever.top_by_facet(
                facet_name, group=group,
                category=None if group else decision.category_hint,
                facets=facet_filter, desc=(direction != "asc"),
            )
            if not ranked and group:  # bad/absent group → rank across the category
                ranked = self._retriever.top_by_facet(
                    facet_name, group=None, category=decision.category_hint,
                    facets=facet_filter, desc=(direction != "asc"),
                )
                if ranked:
                    notes.append(f"top_by_facet group={group} empty → category-wide")
            roster = [f"{t} ({v})" for t, v in ranked]
            if roster:
                notes.append(f"top_by_facet {sort} (group={group}) → {len(roster)}")
        elif group or (facet_filter and not decision.entity_focus):
            # A facet-only roster makes sense for a category question ("dexterity
            # weapons"), but NOT when a specific item is the focus — there, the facet
            # (e.g. dlc=True) would enumerate every DLC item as noise. Entity wins.
            roster = self._retriever.enumerate_titles(
                group, category=decision.category_hint, facets=facet_filter
            )
            if not roster:  # category_hint may over-constrain — retry without it
                roster = self._retriever.enumerate_titles(group, facets=facet_filter)
            if roster:
                notes.append(f"enumerate group={group} facets={facet_filter} → {len(roster)} titles")

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