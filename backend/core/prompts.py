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

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retriever import RetrievedChunk


# ── Context assembly ──────────────────────────────────────────────────────────

def format_context(chunks: list["RetrievedChunk"], max_chars: int = 6000) -> str:
    """
    Format retrieved chunks into a grounded context block. Each chunk is labelled
    with its source (title · section) so the model can attribute facts, and we
    cap total length so the prompt stays within budget.
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
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)


def format_sources(chunks: list["RetrievedChunk"]) -> list[dict]:
    """
    Structured source list for the UI's source cards (deduped by url).
    Returned alongside the answer; not part of the prompt.
    """
    seen: set[str] = set()
    sources: list[dict] = []
    for c in chunks:
        if c.url in seen:
            continue
        seen.add(c.url)
        sources.append({
            "title": c.title,
            "url": c.url,
            "section": c.section_heading,
            "image_url": c.image_url,
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
    "ashes of war", "spells",
)


def _is_visual_page(title: str, category: str) -> bool:
    t = title.lower()
    if any(h in t for h in _NON_VISUAL_TITLE_HINTS):
        return False
    return True


def select_images(
    chunks: list["RetrievedChunk"],
    entity_focus: Optional[list[str]] = None,
    max_images: int = 4,
) -> list[dict]:
    """
    Entity-driven, contribution-ranked image selection.

    The number of images is DYNAMIC — it follows how many distinct relevant
    subject-pages the answer actually draws on. A focused item/boss question
    yields one or two; a sweeping lore answer yields several (the entity, plus
    the other pages it weaves together).

    Relevance bar (tightened after live testing showed tangential images):
      - The resolved-entity page always qualifies.
      - Other pages qualify only if they contributed >= MIN_CONTRIB chunks OR
        ranked highly (best score >= STRONG_SCORE). A single weak tangential
        chunk no longer earns an image slot.

    Order:
      1. The resolved entity's OWN page image(s) first.
      2. Then contributing pages by chunk count, tie-broken by best score.
    Non-visual pages (stats/mechanics/comparison) are skipped. Deduped by image.
    """
    entity_focus = entity_focus or []
    entity_norm = {e.lower() for e in entity_focus}

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

    # Apply the relevance bar: entity pages always qualify; others must clear
    # the contribution or score threshold.
    qualified = [
        p for p in pages.values()
        if p["is_entity"] or p["count"] >= MIN_CONTRIB or p["best"] >= STRONG_SCORE
    ]

    # Rank: entity pages first, then by contribution (count), then by best score.
    ranked = sorted(
        qualified,
        key=lambda p: (p["is_entity"], p["count"], p["best"]),
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
- If the context doesn't contain the answer, say so plainly and suggest what the \
player might ask instead — don't guess.
- Be well-organized and concise. Use short paragraphs or tight lists for steps, \
drops, or locations.
- Your response must be pure answer prose. Do NOT reference context passage \
numbers (e.g. never write "[1]", "passage [2]", "according to context [3]", \
"the context passage you're referring to is [N]", or any similar phrasing). \
The interface shows sources as separate cards — never mention them in your text.
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

def build_messages(
    question: str,
    chunks: list["RetrievedChunk"],
    tone: str = "scholar",
    history: Optional[str] = None,
    cryptic_variant: Optional[str] = None,
) -> list[tuple[str, str]]:
    """
    Assemble the (role, content) message list for the LLM.

    tone: "scholar" | "cryptic"
    cryptic_variant: optionally pass CRYPTIC_STRONG or CRYPTIC_RESTRAINED to
        override the default; ignored for scholar.
    """
    if tone == "cryptic":
        system = cryptic_variant or CRYPTIC_SYSTEM
    else:
        system = SCHOLAR_SYSTEM

    context = format_context(chunks)

    human_parts = []
    if history:
        human_parts.append(f"Recent conversation:\n{history}\n")
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