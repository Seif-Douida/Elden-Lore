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
IntentT    = Literal["drops", "strategy", "location", "dialogue", "lore", "quest"]
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


# ── Call 2 schema ─────────────────────────────────────────────────────────────

class MentionSelection(BaseModel):
    mention: str = Field(description="The user mention being resolved, verbatim.")
    choice_index: int = Field(
        description="1-based index of the chosen candidate from that mention's "
                    "numbered list; 0 if none of the candidates is correct.",
    )


class EntityChoice(BaseModel):
    """
    Index-based selection — the LLM returns the NUMBER of its chosen candidate
    per mention, never a name. This makes it structurally impossible to return
    an off-list/paraphrased entity (the failure we hit with name-based output,
    where the model 'helpfully' returned 'Queen Marika the Eternal' instead of
    an exact candidate). Use 0 to mean "none of these fit".
    """
    selections: list[MentionSelection] = Field(
        default_factory=list,
        description="One entry per mention, with the 1-based index of the best "
                    "candidate (or 0 if none fit).",
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
misspellings. Do not correct them. ("Melania" stays "Melania".)
- needs_retrieval is false only for pure greetings/smalltalk.
- intent: pick the single best fit. "what does X drop" -> drops; "how do I beat \
X" -> strategy; "where is X" -> location; "what does X say" -> dialogue; \
"who is X / story of X" -> lore; "how do I progress X's quest" -> quest.
- tone: 'cryptic' only when the user wants lore/story atmosphere; else 'scholar'.
Be decisive and concise."""

CALL2_SYSTEM = """You disambiguate Elden Ring entity names. For each mention you \
are given a NUMBERED list of candidate names. Choose the single candidate the \
user most likely meant, using the question for context, and return its NUMBER. \
If none of the candidates is correct, return 0 for that mention.

You MUST choose by number from the list provided. Do not write entity names — \
only the index numbers. Example: if candidate 2 is correct, return choice_index 2."""


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
            # Auto-accept: only one candidate, or a clearly dominant top score.
            dominant = (
                len(cands) == 1
                or (top_score >= AUTOACCEPT_SCORE and
                    (len(cands) < 2 or top_score - cands[1][1] >= 10))
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

            # Map each (mention, index) back to the EXACT candidate string.
            accepted, rejected = [], []
            for sel in choice.selections:
                cands = ambiguous.get(sel.mention)
                if cands is None:
                    # mention key drifted; try a tolerant match
                    key = next((k for k in ambiguous if _norm(k) == _norm(sel.mention)), None)
                    cands = ambiguous.get(key) if key else None
                if cands and 1 <= sel.choice_index <= len(cands):
                    accepted.append(cands[sel.choice_index - 1])
                else:
                    rejected.append((sel.mention, sel.choice_index))
            entity_focus.extend(accepted)
            call2_debug = {
                "shortlist_sent": ambiguous,
                "raw_selections": [(s.mention, s.choice_index) for s in choice.selections],
                "accepted": accepted,
                "rejected": rejected,
            }

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