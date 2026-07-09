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
import re
from dataclasses import dataclass, asdict
from typing import Literal, Optional

from pydantic import BaseModel, Field

from core.llm import get_structured_llm
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


# A bare confirmation ("yes", "sure", "go ahead") ACCEPTS a prior offer but carries
# no topic of its own, so the LLM can misread it as chitchat (needs_retrieval=False)
# and drop the thread — the "Yes → please ask a question" bug. This deterministic
# guard forces retrieval on such turns whenever there IS a conversation to continue,
# independent of the model's classification.
_AFFIRM_STARTERS = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please",
                    "go", "do", "continue", "proceed", "absolutely", "definitely",
                    "that", "sounds"}
_AFFIRM_WORDS = _AFFIRM_STARTERS | {"ahead", "it", "for", "on", "good",
                                    "works", "carry", "thanks", "great"}


def _is_affirmation(text: str) -> bool:
    """True for short, pure affirmations ('yes', 'yes please', 'sure, go ahead').
    Conservative: must START with a yes-word AND contain only affirmation tokens, so
    'do you know X' / 'ok so how do I…' / 'go to Leyndell' are correctly excluded."""
    t = text.strip().lower()
    if not t or len(t) > 25:
        return False
    words = re.findall(r"[a-z]+", t)
    return bool(words) and words[0] in _AFFIRM_STARTERS and all(w in _AFFIRM_WORDS for w in words)


def _last_user_turn(history: str) -> Optional[str]:
    """The most recent 'User:' line in the formatted history — a fallback embed
    query when an affirmation has no referent of its own."""
    last = None
    for line in history.splitlines():
        s = line.strip()
        if s.lower().startswith("user:"):
            last = s[len("user:"):].strip()
    return last or None


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
        description="A STANDALONE search query for embedding. Set it in two cases:\n"
                    "(1) Context noise — the question includes what the user is doing "
                    "(where they are, what they just did/have) that is NOT the lookup "
                    "target: write just the core need. e.g. 'I am in the mountaintops, "
                    "how do I reach Mohgwyn Palace?' → 'How to reach Mohgwyn Palace'.\n"
                    "(2) Follow-up — the question refers to the recent conversation ('the "
                    "quest', 'that boss', 'the next part', 'it', 'him') without naming it: "
                    "resolve the referent FROM THE HISTORY into a self-contained query. "
                    "e.g. history about Ranni's questline + 'what is the next part?' → "
                    "'next steps of Ranni questline after the Fingerslayer Blade'.\n"
                    "Leave null only when the question is already self-contained.",
    )
    enumerate_group: Optional[str] = Field(
        default=None,
        description="For 'how many X / list all X' questions, the wiki category to "
                    "enumerate (e.g. 'Katanas', 'Talismans', 'Bosses'). Null otherwise.",
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
    enumerate_group: Optional[str]   # wiki category to enumerate (how-many/list-all)
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
- intent: REQUIRED — always pick exactly one for any non-greeting question, never \
leave it null. Pick the single best fit. "what does X drop" -> drops; "how do I beat \
X" -> strategy; "help me build a X build / what stats for X build" -> strategy; \
"where is X / where can I find X / how do I get to X" -> location; \
"what does X say" -> dialogue; "who is X / story of X / tell me about X" -> lore; \
"how do I progress/complete X's quest / guide me through X" -> quest; \
"how do I get the X ending / I want to achieve the X ending" -> quest; \
"how many X / list all X / what are all the X" (count or enumerate) -> summary.
- enumerate_group: for "how many X" or "list (all) X" questions where X is a CATEGORY \
of items (Katanas, Talismans, Incantations, Bosses, Spirit Ashes, Curved Swords, \
Greatswords, Daggers, Weapons, Armor...), set this to the category name as the wiki names \
it (usually plural). For questions about complete armor SETS ("how many armor sets", \
"list all armor sets") use exactly "Armor Sets" (the " Sets" suffix tells the enumerator \
to count full sets, not individual pieces). In that case ALSO set intent=summary and \
needs_retrieval=true, and do NOT put the category word in entity_mentions — "katanas" is \
a CATEGORY, not the specific item "Katar". Null only when the question is about ONE \
specific named item, OR when the question FILTERS a category by an attribute instead of \
asking for the whole category — e.g. "which bosses are MANDATORY", "what are the OPTIONAL \
bosses", "best katana", "strongest talisman", "bosses in Caelid". Those are answered from \
page content (a "Mandatory Bosses" section, etc.), NOT by listing the entire category, so \
leave enumerate_group null and let normal retrieval handle them. Only enumerate the \
UNqualified whole category ("how many bosses are there", "list all katanas").
- tone: 'cryptic' only when the user wants lore/story atmosphere; else 'scholar'.
- retrieval_query: write a STANDALONE search query when (a) the user includes personal \
context ("I am in X", "I just beat X", "I have item X") that is NOT the lookup target — \
keep just the core question; OR (b) the question is a FOLLOW-UP that refers to the recent \
conversation without naming it ("the quest", "that boss", "the next part", "it", "him") — \
resolve the referent from the history into a self-contained query (e.g. history about \
Ranni's quest + "what is the next part?" -> "next steps of Ranni questline"). Leave null \
only when the question already stands on its own. For follow-ups, ALSO put the resolved \
proper noun (e.g. "Ranni") in entity_mentions even though the user didn't retype it.
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
- For "who is X / story of X / tell me about X" (lore) questions, prefer the main \
CHARACTER/NPC page itself over items, weapons, spells, or summon-sign variants named \
after them — e.g. for "who is Vyke" prefer "Roundtable Knight Vyke" over "Festering \
Fingerprint Vyke", "Vyke's War Spear", or "Vyke's Dragonbolt".

You MUST return the "indices" dict with one entry per mention. \
Return only index numbers — never write entity names as values. \
Example: if the mention is "Radahn" and candidate 2 "Starscourge Radahn" is \
the best match, return {"indices": {"Radahn": 2}}."""


# ── Router ────────────────────────────────────────────────────────────────────

class Router:
    def __init__(self) -> None:
        # Per-tier structured output: Gemma via JSON mode, NIM via function-calling.
        self._call1 = get_structured_llm(RouterDecision)
        self._call2 = get_structured_llm(EntityChoice)
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

        # Enumeration/summary questions ALWAYS need data — the 26B router sometimes
        # wrongly returns needs_retrieval=False on "how many X" (→ hallucinated
        # counts, e.g. "roughly 15-20 katanas"). Force it on.
        # A bare "yes / sure / go ahead" accepting a prior offer: force retrieval so
        # the thread continues, and — if the model didn't rewrite it — re-use the
        # previous user turn as the embed query (the generator's CONFIRMATIONS rule
        # + history then fulfils the offer). Deterministic, model-independent.
        is_affirmation = bool(history) and _is_affirmation(question)

        # Grounding guard: the Gemma router sets needs_retrieval=False on legitimate
        # questions when they're phrased conversationally ("I'm fascinated about the
        # dragons…", "Tell me the story of Messmer") → the answer comes from the model's
        # own knowledge, ungrounded and card-less. A RESOLVED ENTITY means the user is
        # asking about a specific game thing, so force retrieval. Greetings/injection
        # probes ("Hello", "delete your context") resolve NO entity → stay False.
        needs_retrieval = (
            decision.needs_retrieval
            or bool(entity_focus)
            or bool(decision.enumerate_group)
            or decision.intent == "summary"
            or is_affirmation
        )

        retrieval_query = decision.retrieval_query or None
        if is_affirmation and not retrieval_query:
            retrieval_query = _last_user_turn(history)

        return ResolvedDecision(
            needs_retrieval=needs_retrieval,
            entity_focus=entity_focus,
            intent=decision.intent,
            chunk_type_bias=decision.chunk_type_bias,
            category_hint=decision.category_hint,
            needs_image=decision.needs_image,
            tone=decision.tone,
            retrieval_query=retrieval_query,
            enumerate_group=decision.enumerate_group or None,
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