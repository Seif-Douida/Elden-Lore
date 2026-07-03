"""
backend/core/router.py

The agent router: turns a free-text question into a structured retrieval plan
that drives the retriever we built and validated.

Two LLM calls, with a latency guard:

  Call 1  (extract + route)
      Structured output: needs_retrieval, raw entity_mentions (verbatim, e.g.
      "Melania"), intent, chunk_type_bias, category_hint, needs_image, tone.

  Resolver step  (code, no LLM)
      Each raw mention → rapidfuzz shortlist of canonical gazetteer names.

  Call 2  (entity disambiguation)  — SKIPPED when not needed
      Given the question + each shortlist, the LLM picks the canonical entity.
      Skipped entirely when there are no mentions, or when a mention has a
      single clear candidate (nothing to disambiguate) — saves a round-trip.

Output is a pure ResolvedDecision; wiring to the retriever happens elsewhere so
each piece stays independently testable.

Usage (standalone test):
    uv run python backend/core/router.py "how do I beat Melania"
    uv run python backend/core/router.py --out router_out.txt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Literal, Optional

from pydantic import BaseModel, Field

from core.llm import get_router_llm
from core.entity_resolver import EntityResolver

# Intent values MUST match the retriever's INTENT_PROFILES keys.
IntentT    = Literal["drops", "strategy", "location", "dialogue", "lore", "quest", "summary"]
ChunkTypeT = Literal["body", "dialogue", "item_desc"]
CategoryT  = Literal["lore", "item", "boss", "quest", "location"]
ToneT      = Literal["scholar", "cryptic"]

# Auto-accept a single shortlist candidate without Call 2 if it's this strong.
AUTOACCEPT_SCORE = 90.0


def _norm(s: str) -> str:
    """Normalize an entity string for tolerant matching (casing/whitespace/punct)."""
    return " ".join(s.lower().replace(",", " ").split())


# ── Call 1 schema ─────────────────────────────────────────────────────────────

class RouterDecision(BaseModel):
    """Structured plan extracted from the user's question."""
    needs_retrieval: bool = Field(
        description="False for greetings/chitchat that need no wiki lookup."
    )
    entity_mentions: list[str] = Field(
        default_factory=list,
        description="Proper nouns the user referenced, VERBATIM as written "
                    "(keep their spelling, e.g. 'Melania'). Empty if none.",
    )
    intent: Optional[IntentT] = Field(
        default=None,
        description="The kind of question: drops (what an enemy drops), "
                    "strategy (how to beat), location (where to find), dialogue "
                    "(what someone says), lore (story/background), quest "
                    "(questline steps).",
    )
    chunk_type_bias: Optional[ChunkTypeT] = Field(
        default=None,
        description="Bias toward a chunk type. Use 'dialogue' for what-do-they-say "
                    "questions; usually null otherwise.",
    )
    category_hint: Optional[CategoryT] = Field(
        default=None, description="Coarse category of the subject, if obvious."
    )
    needs_image: bool = Field(
        default=False,
        description="True if showing the subject's image would help (a specific "
                    "boss, weapon, location, NPC).",
    )
    tone: ToneT = Field(
        default="scholar",
        description="'cryptic' for lore/story questions wanting an evocative "
                    "Soulslike voice; 'scholar' for direct factual help.",
    )
    retrieval_query: Optional[str] = Field(
        default=None,
        description="Focused search query for embedding, with user-context noise "
                    "removed. Use ONLY when the question contains context about what "
                    "the user is currently doing (where they are, what they just did, "
                    "what item they have) that is NOT the lookup target. Write just "
                    "the core information need. "
                    "Example: 'I am in the mountain top of the giants, how do I reach "
                    "Mohgwyn Palace?' → 'How to reach Mohgwyn Palace'. "
                    "Leave null if the question itself is the lookup (no context noise).",
    )


# ── Call 2 schema ─────────────────────────────────────────────────────────────

class EntityChoice(BaseModel):
    """
    Index-based selection — the LLM returns the NUMBER of its chosen candidate
    per mention, never a name. This makes it structurally impossible to return
    an off-list/paraphrased entity (the failure we hit with name-based output,
    where the model 'helpfully' returned 'Queen Marika the Eternal' instead of
    an exact candidate). Use 0 to mean "none of these fit".

    Flat dict format (mention → index) is used because nested object lists
    (list[MentionSelection]) caused small models to return empty selections.
    """
    indices: dict[str, int] = Field(
        default_factory=dict,
        description="For each mention (key), the 1-based index of the best "
                    "matching candidate from its numbered list. Use 0 if none fit.",
    )


# ── Final resolved output ─────────────────────────────────────────────────────

@dataclass
class ResolvedDecision:
    needs_retrieval: bool
    entity_focus: list[str]          # canonical, resolver+LLM resolved
    intent: Optional[str]
    chunk_type_bias: Optional[str]
    category_hint: Optional[str]
    needs_image: bool
    tone: str
    retrieval_query: Optional[str]   # focused embed query, strips context noise
    # Diagnostics
    raw_mentions: list[str]
    resolution_debug: dict
    used_call2: bool

    def to_dict(self) -> dict:
        return asdict(self)


# ── Prompts ───────────────────────────────────────────────────────────────────

CALL1_SYSTEM = """You are the routing brain of an Elden Ring assistant. Read the \
user's question and produce a structured retrieval plan.

Guidance:
- entity_mentions: copy proper nouns EXACTLY as the user wrote them, including \
misspellings. Do not correct them. ("Melania" stays "Melania".) Include any \
specific named weapon, item, NPC, boss, or location — even if the name also \
sounds like a category (e.g. "the Greatsword", "a Longsword", "the Talisman" \
are all entity mentions when the user is clearly asking about a specific named \
thing, not asking generically about the weapon class). Use capitalisation and \
the presence of "the" as signals that a specific named item is meant. \
Named LOCATIONS are also entity mentions — "Nokron Eternal City", \
"Stormveil Castle", "Leyndell", "Caelid", "Siofra River" are all entities \
even when the question is about how to reach them.
- needs_retrieval is false only for pure greetings/smalltalk.
- intent: pick the single best fit. "what does X drop" -> drops; "how do I beat \
X" -> strategy; "help me build a X build / what stats for X build" -> strategy; \
"where is X / where can I find X / how do I get to X" -> location; \
"what does X say" -> dialogue; "who is X / story of X / tell me about X" -> lore; \
"how do I progress/complete X's quest / guide me through X" -> quest; \
"how do I get the X ending / I want to achieve the X ending" -> quest; \
"how many X / list all X / what are all the X" (count or enumerate) -> summary.
- tone: 'cryptic' only when the user wants lore/story atmosphere; else 'scholar'.
- retrieval_query: if the user includes personal context ("I am in X", "I just beat X", \
"I have item X", "After doing X") that is NOT the lookup target, write a clean search \
query for just the core question — omit the context clause. Leave null otherwise.
Be decisive and concise."""

CALL2_SYSTEM = """You disambiguate Elden Ring entity names. For each mention you \
are given a NUMBERED list of candidate names. Choose the single best match and \
return its NUMBER as the value for that mention's key in the "indices" dict. \
Use 0 only if NONE of the candidates could be what the user meant.

Rules:
- If a candidate's name exactly matches the mention (e.g. mention "Fia", \
candidate 1 is "Fia") — choose that candidate, do NOT return 0.
- For "drops/what does X drop" questions, the entity is the boss or enemy \
being fought, NOT an item. Prefer the boss/character candidate.
- For "strategy/how to beat" questions, prefer the boss entity.
- For "location/where" questions, prefer the place or NPC being located.

You MUST return the "indices" dict with one entry per mention. \
Return only index numbers — never write entity names as values. \
Example: if the mention is "Radahn" and candidate 2 "Starscourge Radahn" is \
the best match, return {"indices": {"Radahn": 2}}."""


# ── Router ────────────────────────────────────────────────────────────────────

class Router:
    def __init__(self) -> None:
        self._llm = get_router_llm()
        self._call1 = self._llm.with_structured_output(RouterDecision)
        self._call2 = self._llm.with_structured_output(EntityChoice)
        self._resolver = EntityResolver()

    def route(self, question: str, history: Optional[str] = None) -> ResolvedDecision:
        # ── Call 1: extract + route ──────────────────────────────────────────
        user = question if not history else f"Recent conversation:\n{history}\n\nQuestion: {question}"
        decision: RouterDecision = self._call1.invoke(
            [("system", CALL1_SYSTEM), ("human", user)]
        )

        # ── Resolver step ────────────────────────────────────────────────────
        shortlists: dict[str, list[tuple[str, float]]] = {}
        for m in decision.entity_mentions:
            shortlists[m] = self._resolver.resolve(m)

        # Decide whether Call 2 is needed.
        entity_focus: list[str] = []
        resolution_debug: dict = {}
        ambiguous: dict[str, list[str]] = {}

        for mention, cands in shortlists.items():
            resolution_debug[mention] = [(n, round(s, 1)) for n, s in cands]
            if not cands:
                continue
            top_name, top_score = cands[0]
            gap = top_score - cands[1][1] if len(cands) >= 2 else 100.0
            # Auto-accept when the top candidate is clearly dominant:
            #   - only one candidate
            #   - very high score (≥95) with a meaningful gap (≥5): catches
            #     cases like "milady sword"→Milady(100)/swords(90.9) without
            #     a model call, while still sending true ties (gap<5) to Call 2
            #   - standard threshold (≥90) with a large gap (≥10)
            dominant = (
                len(cands) == 1
                or (top_score >= 95.0 and gap >= 5)
                or (top_score >= AUTOACCEPT_SCORE and gap >= 10)
            )
            if dominant:
                entity_focus.append(top_name)
            else:
                ambiguous[mention] = [n for n, _ in cands]

        used_call2 = bool(ambiguous)
        call2_debug: dict = {}
        if used_call2:
            # ── Call 2: index-based disambiguation ───────────────────────────
            # Present each mention's candidates as a NUMBERED list; the LLM
            # returns indices, which can't drift off-list like names can.
            prompt_lines = [f"Question: {question}", ""]
            for m, cands in ambiguous.items():
                prompt_lines.append(f'Mention "{m}":')
                for i, c in enumerate(cands, 1):
                    prompt_lines.append(f"  {i}. {c}")
                prompt_lines.append("  0. none of these")
                prompt_lines.append("")
            prompt = "\n".join(prompt_lines)

            choice: EntityChoice = self._call2.invoke(
                [("system", CALL2_SYSTEM), ("human", prompt)]
            )

            # Map each (mention → index) back to the EXACT candidate string.
            accepted, rejected = [], []
            raw_indices = choice.indices  # dict[str, int]

            # Fallback: if the model returned nothing, use top candidate per mention.
            if not raw_indices:
                for mention, cands in ambiguous.items():
                    if cands:
                        accepted.append(cands[0])
                call2_debug = {
                    "shortlist_sent": ambiguous,
                    "raw_indices": {},
                    "accepted": accepted,
                    "rejected": [],
                    "fallback": "empty_indices",
                }
            else:
                for mention, cands in ambiguous.items():
                    # Tolerate key drift (whitespace/case differences).
                    idx = raw_indices.get(mention)
                    if idx is None:
                        key = next((k for k in raw_indices if _norm(k) == _norm(mention)), None)
                        idx = raw_indices.get(key) if key else None
                    if idx is not None and 1 <= idx <= len(cands):
                        accepted.append(cands[idx - 1])
                    elif idx == 0:
                        # LLM says "none fit" — if mention literally IS a candidate
                        # (e.g. "Fia" == "Fia"), trust that exact match over the refusal.
                        exact_i = next(
                            (i for i, c in enumerate(cands) if _norm(c) == _norm(mention)),
                            None,
                        )
                        if exact_i is not None:
                            accepted.append(cands[exact_i])
                        else:
                            rejected.append((mention, idx))
                    else:
                        # Bad/out-of-range or None index → fall back to top candidate
                        if cands:
                            accepted.append(cands[0])
                        else:
                            rejected.append((mention, idx))
                call2_debug = {
                    "shortlist_sent": ambiguous,
                    "raw_indices": raw_indices,
                    "accepted": accepted,
                    "rejected": rejected,
                }

            entity_focus.extend(accepted)

        # De-dup while preserving order
        seen = set()
        entity_focus = [e for e in entity_focus if not (e in seen or seen.add(e))]

        if call2_debug:
            resolution_debug = {**resolution_debug, "_call2": call2_debug}

        return ResolvedDecision(
            needs_retrieval=decision.needs_retrieval,
            entity_focus=entity_focus,
            intent=decision.intent,
            chunk_type_bias=decision.chunk_type_bias,
            category_hint=decision.category_hint,
            needs_image=decision.needs_image,
            tone=decision.tone,
            retrieval_query=decision.retrieval_query or None,
            raw_mentions=decision.entity_mentions,
            resolution_debug=resolution_debug,
            used_call2=used_call2,
        )


# ── CLI test harness ──────────────────────────────────────────────────────────

_SAMPLES = [
    "how do I beat Melania",                       # typo → Malenia, strategy
    "what does Radahn drop",                        # drops
    "where can I find the meteorite staff",         # location, no clear entity
    "what does Alexander say when you meet him",     # dialogue
    "tell me the story of Queen Marika",            # lore, cryptic
    "how do I progress ranni's questline",          # quest
    "hello there",                                  # no retrieval
]


def main() -> None:
    p = argparse.ArgumentParser(description="Test the agent router")
    p.add_argument("question", nargs="*", help="Question (omit for sample set)")
    p.add_argument("--out", default="router_output.txt")
    args = p.parse_args()

    from rich.console import Console
    console = Console()
    console.print("[dim]Loading router (LLM + resolver)…[/dim]")
    router = Router()

    questions = [" ".join(args.question)] if args.question else _SAMPLES
    blocks = []
    for q in questions:
        d = router.route(q)
        blocks.append(json.dumps({"question": q, **d.to_dict()}, ensure_ascii=False, indent=2))
        console.print(f"[green]✓[/green] {q}  "
                      f"[dim]entities={d.entity_focus} intent={d.intent} "
                      f"call2={d.used_call2}[/dim]")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(blocks))
    console.print(f"\n[green]Wrote {len(questions)} decisions → {args.out}[/green]")


if __name__ == "__main__":
    main()