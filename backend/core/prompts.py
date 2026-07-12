"""
backend/core/prompts.py

Prompt assembly for the generation layer: turns a PipelineResult (chunks +
routing decision) into the message list the LLM answers from.

Two tones (the router picks which):
  - Scholar : clear, accurate, well-organized guide. STRICTLY grounded — answers
    only from the retrieved context, and says so when the context lacks the
    answer. For factual questions (drops, locations, strategy, quests).
  - Cryptic : the Soulslike/VaatiVidya lore voice. Evocative and interpretive —
    free to CONNECT and narrate the retrieved fragments into atmosphere, but its
    FACTS stay anchored to the context (interpretation, not fabrication). For
    lore/story questions.

Citations are NOT inline — the answer reads cleanly and the UI renders source
cards from each chunk's url/title/image_url. The prompts therefore ask for
grounded prose, not bracketed citations.

Two Cryptic intensities are provided (CRYPTIC_STRONG / CRYPTIC_RESTRAINED) to
choose between.
"""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retriever import RetrievedChunk


# ── Context assembly ──────────────────────────────────────────────────────────

def format_context(chunks: list["RetrievedChunk"], max_chars: int = 12000) -> str:
    """
    Format retrieved chunks into a grounded context block. Each chunk is labelled
    with its source (title · section) so the model can attribute facts, and we
    cap total length so the prompt stays within budget.

    We `continue` (not `break`) past an over-budget chunk so one long early chunk
    doesn't drop all the later (often more on-point) ones — keep packing what fits.
    Gemma's context window easily absorbs 12k chars.
    """
    if not chunks:
        return "(no relevant wiki passages were retrieved)"

    blocks: list[str] = []
    total = 0
    for c in chunks:
        src = c.title
        if c.section_heading and c.section_heading.lower() != "overview":
            src += f" · {c.section_heading}"
        block = f"=== {src} ===\n{c.raw_text}"
        if total + len(block) > max_chars:
            continue
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def _norm_text(s: str) -> str:
    """Lowercase, punctuation→space, collapse whitespace — for tolerant matching."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split())


def _named_in_answer(title: str, answer_norm: str) -> bool:
    """Is this page's title mentioned in the (normalized) answer text?"""
    t = _norm_text(title)
    return bool(t) and t in answer_norm


def format_sources(
    chunks: list["RetrievedChunk"],
    entity_focus: Optional[list[str]] = None,
    answer_text: Optional[str] = None,
) -> list[dict]:
    """
    Structured source list for the UI's source cards (deduped by url).

    When `answer_text` is given, gate cards to pages the answer actually draws on
    — the resolved entity page(s), or any page whose title is named in the answer
    — so refusals/enumerations don't spray tangential cards. Without answer_text
    (e.g. legacy callers) it returns one card per retrieved page as before.
    """
    entity_norm = {e.lower() for e in (entity_focus or [])}
    answer_norm = _norm_text(answer_text) if answer_text else ""

    seen: set[str] = set()
    sources: list[dict] = []
    for c in chunks:
        if c.url in seen:
            continue
        if answer_norm:
            keep = (c.title.lower() in entity_norm) or _named_in_answer(c.title, answer_norm)
            if not keep:
                continue
        seen.add(c.url)
        sources.append({
            "title": c.title,
            "url": c.url,
            "section": c.section_heading,
            "image_url": c.image_url or None,   # guard empty string → no broken img slot
            "category": c.category,
        })
    return sources


# Pages that are about mechanics/stats/comparisons rather than a visual subject —
# their images (if any) are charts or generic banners, not worth showing.
_NON_VISUAL_TITLE_HINTS = (
    "stats", "comparison", "calculating", "motion values", "damage",
    "status effects", "poise", "hemorrhage", "skills", "level", "hp",
    "upgrades", "patch notes", "controls", "builds",
    "walkthrough", "route", "progress route", "game progress",
    # Category/index hub pages — images are generic wiki graphics, not visual subjects
    "bosses", "items", "locations", "weapons", "npcs", "npc",
    "armor", "shields", "sorceries", "incantations", "talismans",
    "ashes of war", "spells", "endings", "great runes", "spirit ashes",
    "crystal tears", "cookbooks", "creatures", "merchants",
    # Weapon/magic SUB-category hubs (plural forms; individual items use singular
    # titles, so these won't false-match e.g. "Knight's Greatsword").
    "daggers", "straight swords", "greatswords", "colossal swords",
    "colossal weapons", "thrusting swords", "heavy thrusting swords",
    "curved swords", "curved greatswords", "katanas", "twinblades",
    "axes", "greataxes", "hammers", "great hammers", "flails",
    "spears", "great spears", "halberds", "reapers", "whips",
    "fists", "bows", "light bows", "greatbows", "crossbows",
    "ballistae", "glintstone staffs", "glintstone staves", "sacred seals",
    "torches", "throwing blades",
)


def _is_visual_page(title: str, category: str) -> bool:
    t = title.lower()
    if any(h in t for h in _NON_VISUAL_TITLE_HINTS):
        return False
    return True


def select_images(
    chunks: list["RetrievedChunk"],
    entity_focus: Optional[list[str]] = None,
    answer_text: Optional[str] = None,
    max_images: int = 4,
) -> list[dict]:
    """
    Answer-gated, contribution-ranked image selection.

    The number of images is DYNAMIC — it follows how many distinct relevant
    subject-pages the answer actually draws on. A focused item/boss question
    yields one or two; a sweeping lore answer yields several (the entity, plus
    the other pages it weaves together).

    Relevance bar (tightened after live testing showed tangential images on
    refusals/enumerations):
      - The resolved-entity page always qualifies (it's the subject).
      - Other pages qualify only if they contributed >= MIN_CONTRIB chunks OR
        ranked highly (best score >= STRONG_SCORE) AND their title is actually
        NAMED in the answer. This drops tangential pages the answer never talks
        about (e.g. Ghostflame Dragon on a smithing-stone count) and clears
        images on refusals (which name no wiki page).

    NOTE: the router's `needs_image` flag is deliberately NOT used as a gate — the
    8B router sets it unreliably (it returned False for "where is Moonveil?"), and
    hard-suppressing on it dropped legit entity images. Answer-mention gating is
    the robust filter instead.

    Order:
      1. The resolved entity's OWN page image(s) first.
      2. Then contributing pages by chunk count, tie-broken by best score.
    Non-visual pages (stats/mechanics/comparison) are skipped. Deduped by image.
    """
    entity_focus = entity_focus or []
    entity_norm = {e.lower() for e in entity_focus}
    answer_norm = _norm_text(answer_text) if answer_text else ""

    MIN_CONTRIB = 2       # a non-entity page needs >=2 chunks to earn an image
    STRONG_SCORE = 0.74   # ...unless a single chunk scored this strongly

    # Aggregate per source page.
    pages: dict[str, dict] = {}
    for c in chunks:
        if not c.image_url:
            continue
        if not _is_visual_page(c.title, c.category):
            continue
        p = pages.setdefault(c.url, {
            "image_url": c.image_url,
            "title": c.title,
            "url": c.url,
            "count": 0,
            "best": 0.0,
            "is_entity": c.title.lower() in entity_norm,
        })
        p["count"] += 1
        p["best"] = max(p["best"], c.final_score if c.boost else c.score)

    # Apply the relevance bar: entity pages always qualify; other pages must
    # clear the contribution/score threshold AND (when we know the answer) be
    # named in it, so tangential pages and refusals don't attach images.
    def _qualifies(p: dict) -> bool:
        if p["is_entity"]:
            return True
        if not (p["count"] >= MIN_CONTRIB or p["best"] >= STRONG_SCORE):
            return False
        if answer_norm and not _named_in_answer(p["title"], answer_norm):
            return False
        return True

    qualified = [p for p in pages.values() if _qualifies(p)]

    # Rank by SPECIFICITY, not volume: entity page first, then pages actually
    # named in the answer, then best score, then chunk count last. (Previously
    # count outranked score, so a generic hub that merely contributed more chunks
    # — e.g. the "Endings" page — beat the specific on-topic page.)
    def _named(p: dict) -> bool:
        return bool(answer_norm) and _named_in_answer(p["title"], answer_norm)
    ranked = sorted(
        qualified,
        key=lambda p: (p["is_entity"], _named(p), p["best"], p["count"]),
        reverse=True,
    )

    # Dedup by image_url (different pages occasionally share an image) and cap.
    out: list[dict] = []
    seen_img: set[str] = set()
    for p in ranked:
        if p["image_url"] in seen_img:
            continue
        seen_img.add(p["image_url"])
        out.append({"image_url": p["image_url"], "title": p["title"], "url": p["url"]})
        if len(out) >= max_images:
            break
    return out


# ── Scholar tone ──────────────────────────────────────────────────────────────

SCHOLAR_SYSTEM = """You are a knowledgeable guide to Elden Ring — precise, clear, \
and helpful, like a seasoned scholar of the Lands Between.

Answer the player's question using ONLY the information in the provided context \
passages. Rules:
- Ground every factual claim in the context. Do not invent items, locations, \
stats, or steps that aren't supported there.
- If the answer genuinely isn't available, say so IN-WORLD by naming the missing \
thing — e.g. "The wiki doesn't note a recommended level for Radahn" — then suggest \
what the player might ask instead; don't guess. NEVER phrase this as "the provided \
information/records/context does not specify/state/contain…" (see the source-material \
rule below), and never open with such a disclaimer before an answer you CAN give.
- Be well-organized and concise. Use short paragraphs or tight lists for steps, \
drops, or locations.
- ENUMERATION & RANKED LISTS: when the context includes a "COMPLETE LIST" or "RANKED \
LIST" block, THAT block is the authoritative answer — use it (not the reference \
passages) for the count, the list, or the ranking, and never say the answer "isn't \
specified" when the block supplies it. For a COMPLETE LIST: state the EXACT total and \
list ~15 examples; if the total exceeds what's shown, point the player to the wiki or \
invite them to ask about a specific item (never dump a huge list, never invent items \
to pad it). (e.g. "There are 108 talismans; notable ones include … — want the full \
list on the wiki?") For a RANKED LIST (best first): the FIRST entry is the answer to a \
superlative ("heaviest/lightest/cheapest/strongest") — name it with its value, then a \
few runners-up (e.g. "The heaviest armor is the Leyndell Knight Set at 28.1, followed \
by Bull-Goat Armor (26.5)…").
- COUNTING FROM A LIST: if the player asks "how many" and there is no explicit total \
in the context, but the context DOES lay out the specific instances or locations (e.g. \
a "Where to Find" list of spots where an item is found), count those and give the \
number — make clear it's the count of locations/instances listed in the wiki (e.g. \
"The wiki lists 6 places to find a Larval Tear in the base game, plus more in the DLC."). \
Only count what's actually listed; don't invent entries to reach a rounder number. \
- GUIDES: for multi-step content (questlines, endings), give the full ordered \
sequence the context supports. If the context only covers part of it, say so plainly \
and invite the player to ask for the next steps.
- FOCUS: answer what was asked directly. Note prerequisites or caveats briefly, but \
do not dwell on tangential conditions at the expense of the main answer.
- For broad questline or overview questions, you may close by offering a focused next \
step — e.g. "Would you like the full step-by-step walkthrough for X?"
- CONFIRMATIONS: if the player's message is a brief affirmation ("Yes", "sure", "ok", \
"please", "go ahead") that accepts an offer or question from the recent conversation, \
treat it as a request to fulfil that offer using the retrieved context — do NOT reply \
that they haven't asked a question.
- Your response must be pure answer prose. Do NOT reference context passage \
numbers (e.g. never write "[1]", "passage [2]", "according to context [3]", \
"the context passage you're referring to is [N]", or any similar phrasing). \
The interface shows sources as separate cards — never mention them in your text.
- NEVER refer to your source material, in ANY wording. This bans "the provided \
context/text/information/records/data", "the documentation", "the context", "the \
available information", "the wiki does not provide/specify a ranked list", and every \
close variant — ESPECIALLY the pattern "the provided … does not specify / state / \
contain / mention …". The player only sees your answer, so these read as robotic. \
Lead with what IS known, stated as settled fact. Only flag a gap when the player \
asked for a specific detail that is genuinely absent, and then name the missing \
THING in-world — "The wiki doesn't note a recommended level for Radahn", "I don't \
have the exact number, but the known locations are…" — never by pointing at \
"information / records / context" as a source, and never as a lead-in before an \
answer you can actually give.
- Do NOT write image placeholders, image captions, or any phrase about where \
an image appears (e.g. never write "[Image of X appears here]", \
"[image appears here]", "see image below", etc.). The interface handles image \
display entirely automatically — your answer text must contain only prose, \
with no image annotations of any kind.
- Stay in a clear, informative register — no purple prose."""


# ── Cryptic tone — two intensities to choose from ─────────────────────────────

CRYPTIC_STRONG = """You are a sage of the Lands Between — your voice that of a \
loremaster who speaks in the hushed, evocative cadence of the Soulslike \
tradition. You address the Tarnished directly, with gravity and mystery.

Weave the provided context passages into lore. Rules:
- Your FACTS come only from the context — names, deeds, relationships, events. \
Do not fabricate lore that isn't grounded there. But you may CONNECT those \
fragments, draw out their implications, and narrate them as a story.
- Speak evocatively: measured, a touch archaic, reverent toward the mythic \
weight of it all. You may open with address such as "Ah, Tarnished…" when it \
fits.
- Embrace the fragmentary nature of this lore — gesture at mystery where the \
context leaves gaps, rather than inventing answers.
- Do NOT reference context passage numbers (e.g. never write "[1]", "passage \
[2]", or "the context passage you're referring to is [N]"). The interface shows \
sources as separate cards — never mention them in your text.
- Do NOT write image placeholders or annotations of any kind (e.g. never write \
"[Image of X appears here]"). The interface displays images automatically.
- If the context holds nothing relevant, say so in voice — admit the threads of \
this tale lie beyond your sight."""

CRYPTIC_RESTRAINED = """You are a loremaster of the Lands Between, recounting its \
history with an evocative, atmospheric voice — measured and a little mysterious, \
but without theatrical roleplay or direct address.

Weave the provided context passages into a flowing account of the lore. Rules:
- Your FACTS come only from the context — names, deeds, relationships, events. \
Do not fabricate lore that isn't grounded there. You may CONNECT fragments and \
draw out their meaning, but not invent.
- Write atmospherically and evocatively, but in measured prose — no "Ah, \
Tarnished", no roleplay framing, no excessive archaism. The mystery comes from \
the lore itself, not from affectation.
- Where the context leaves gaps, acknowledge the uncertainty rather than filling \
it with invention.
- Do NOT reference context passage numbers (e.g. never write "[1]", "passage \
[2]", or "the context passage you're referring to is [N]"). The interface shows \
sources as separate cards — never mention them in your text.
- Do NOT write image placeholders or annotations of any kind (e.g. never write \
"[Image of X appears here]"). The interface displays images automatically.
- If the context holds nothing relevant, say so plainly but in keeping with the \
reflective tone."""

# Default Cryptic variant the assembler uses (swap after you choose).
CRYPTIC_SYSTEM = CRYPTIC_RESTRAINED


# ── Message assembly ──────────────────────────────────────────────────────────

def format_roster(group: str, titles: list[str], max_show: int = 15) -> str:
    """A precise roster (from Qdrant metadata) for enumeration / relational / ranked
    questions: the authoritative answer set the model must use instead of deriving a
    list from the passages (or hallucinating). Entries like "Bull-Goat Armor (26.5)"
    mark a RANKED list (superlatives), where the first entry is the answer."""
    import re
    n = len(titles)
    shown = titles[:max_show]
    ranked = any(re.search(r"\([\d.]+\)\s*$", t) for t in shown)
    if ranked:
        return (f"RANKED LIST — {group}, best first ({n} total). The FIRST entry is "
                f"the answer to a 'most/least/-est' question. Top {len(shown)}: "
                f"{'; '.join(shown)}.")
    return (f"COMPLETE LIST — {group}: {n} total. "
            f"Examples ({len(shown)} of {n}): {', '.join(shown)}.")


def build_messages(
    question: str,
    chunks: list["RetrievedChunk"],
    tone: str = "scholar",
    history: Optional[str] = None,
    cryptic_variant: Optional[str] = None,
    enumerate_group: Optional[str] = None,
    roster: Optional[list[str]] = None,
) -> list[tuple[str, str]]:
    """
    Assemble the (role, content) message list for the LLM.

    tone: "scholar" | "cryptic"
    cryptic_variant: optionally pass CRYPTIC_STRONG or CRYPTIC_RESTRAINED to
        override the default; ignored for scholar.
    roster: for enumeration questions, the metadata-derived list of titles under
        `enumerate_group` — injected as an authoritative COMPLETE LIST block.
    """
    if tone == "cryptic":
        system = cryptic_variant or CRYPTIC_SYSTEM
    else:
        system = SCHOLAR_SYSTEM

    context = format_context(chunks)

    human_parts = []
    if history:
        human_parts.append(f"Recent conversation:\n{history}\n")
    if roster:
        human_parts.append(format_roster(enumerate_group or "Matching results", roster) + "\n")
    human_parts.append(f"Wiki reference material:\n{context}\n")
    human_parts.append(f"Player's question: {question}")
    human = "\n".join(human_parts)

    return [("system", system), ("human", human)]


# ── Demo (no LLM call) — shows the assembled prompt for inspection ────────────

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class _Stub:
        title: str
        section_heading: str
        raw_text: str
        url: str = "https://example.com"
        image_url: Optional[str] = None
        category: str = "lore"

    stub_chunks = [
        _Stub("Starscourge Radahn", "Combat information",
              "Drops 70,000 runes, Remembrance of the Starscourge, Radahn's Great Rune."),
        _Stub("Starscourge Radahn", "Overview",
              "Radahn is a demigod who held back the stars with his gravity magic."),
    ]

    print("=" * 70)
    print("SCHOLAR PROMPT")
    print("=" * 70)
    for role, content in build_messages("What does Radahn drop?", stub_chunks, "scholar"):
        print(f"\n### {role.upper()}\n{content}")

    print("\n" + "=" * 70)
    print("CRYPTIC (RESTRAINED) PROMPT")
    print("=" * 70)
    for role, content in build_messages("Tell me of Radahn", stub_chunks, "cryptic"):
        print(f"\n### {role.upper()}\n{content}")