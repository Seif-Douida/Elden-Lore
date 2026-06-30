"""
backend/core/entity_resolver.py

Resolves a user's (possibly misspelled / partial) entity mention to canonical
entity names from our gazetteer — e.g. "Melania" → "Malenia", "Marika" →
"Queen Marika", "bloodhounds fang" → "Bloodhound's Fang".

Design (validated against real failure cases):
  - A SINGLE fuzzy score is not enough (naive ratio ranked "Enia" above
    "Malenia" for the typo "Melania"). So we combine several rapidfuzz scorers
    and take the best, then break near-ties by corpus frequency.
  - We return the TOP-N shortlist, not a single guess. The router LLM — which
    has the full question for context — makes the final pick. Fuzzy matching
    only needs to get the right entity *into* the shortlist (recall); the LLM
    supplies precision. This is the hybrid the design calls for.

The gazetteer (name → frequency) is built offline by build_gazetteer.py and
cached to data/gazetteer.json.

Usage (standalone test):
    uv run python backend/core/entity_resolver.py "Melania" "marika" "godric"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

GAZETTEER_FILE = Path(__file__).parent / "data" / "gazetteer.json"

DEFAULT_SHORTLIST   = 5
SCORE_FLOOR         = 60.0   # below this, treat as "no confident match"
FREQ_TIEBREAK_BAND  = 5.0    # scores within this many points count as a tie

# Phonetic matching catches homophone typos that edit-distance misses, e.g.
# "Melania" is edit-closer to "Melina" than to "Malenia", but all three are
# phonetically identical — so a phonetic score lifts "Malenia" into the
# shortlist where the router LLM can pick it using question context.
try:
    import jellyfish  # noqa: F401
    _HAVE_JELLYFISH = True
except Exception:
    _HAVE_JELLYFISH = False


def _phonetic_key(s: str) -> str:
    """Metaphone if available; else a crude vowel-stripped consonant skeleton."""
    if _HAVE_JELLYFISH:
        try:
            return jellyfish.metaphone(s)
        except Exception:
            pass
    import re
    s = s.lower()
    if not s:
        return ""
    first, rest = s[0], re.sub(r"[aeiou]", "", s[1:])
    return re.sub(r"(.)\1+", r"\1", first + rest)


class EntityResolver:
    def __init__(self, gazetteer_path: Path = GAZETTEER_FILE) -> None:
        if not gazetteer_path.exists():
            raise FileNotFoundError(
                f"No gazetteer at {gazetteer_path}. Run build_gazetteer.py first."
            )
        data = json.loads(gazetteer_path.read_text(encoding="utf-8"))
        self._freq: dict[str, int] = data["entities"]
        self._names: list[str] = list(self._freq.keys())
        # Precompute phonetic keys once for the whole gazetteer.
        self._phon: dict[str, str] = {n: _phonetic_key(n) for n in self._names}

    # ── core resolution ──────────────────────────────────────────────────────
    def resolve(self, mention: str, shortlist: int = DEFAULT_SHORTLIST
                ) -> list[tuple[str, float]]:
        """
        Return up to `shortlist` (canonical_name, score) candidates, best first.
        Empty list if nothing clears SCORE_FLOOR.
        """
        from rapidfuzz import fuzz

        mention = mention.strip()
        if not mention:
            return []

        mention_phon = _phonetic_key(mention)

        # Combine scorers — different ones catch different error modes:
        #   WRatio           → general weighted ratio (typos, casing)
        #   token_sort_ratio → word-order / multi-word names
        #   partial_ratio    → substring ("Marika" in "Queen Marika")
        #   phonetic         → homophone typos ("Melania" ~ "Malenia")
        scored: list[tuple[str, float]] = []
        for name in self._names:
            phon_score = fuzz.ratio(mention_phon, self._phon[name]) if mention_phon else 0.0
            s = max(
                fuzz.WRatio(mention, name),
                fuzz.token_sort_ratio(mention, name),
                fuzz.partial_ratio(mention.lower(), name.lower()),
                phon_score,
            )
            scored.append((name, s))

        # Sort by score, then by corpus frequency for near-ties (prefer the
        # entity that actually appears most as a filterable value).
        scored.sort(key=lambda x: (x[1], self._freq.get(x[0], 0)), reverse=True)

        # Frequency tiebreak within the top band: if the runner-up is within
        # FREQ_TIEBREAK_BAND of the leader but far more frequent, promote it.
        if len(scored) >= 2:
            top_score = scored[0][1]
            band = [x for x in scored if top_score - x[1] <= FREQ_TIEBREAK_BAND]
            band.sort(key=lambda x: self._freq.get(x[0], 0), reverse=True)
            # stitch the reordered band back on top
            rest = [x for x in scored if x not in band]
            scored = band + rest

        out = [(n, float(s)) for n, s in scored if s >= SCORE_FLOOR][:shortlist]
        return out

    def best(self, mention: str) -> Optional[str]:
        """Single best canonical name, or None if no confident match."""
        r = self.resolve(mention, shortlist=1)
        return r[0][0] if r else None

    def resolve_many(self, mentions: list[str], shortlist: int = DEFAULT_SHORTLIST
                     ) -> dict[str, list[tuple[str, float]]]:
        return {m: self.resolve(m, shortlist) for m in mentions}


# ── CLI test ──────────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    from rich.console import Console
    console = Console()

    mentions = sys.argv[1:] or ["Melania", "Malenia", "marika", "godric",
                                 "bloodhounds fang", "ranni", "radan"]
    try:
        resolver = EntityResolver()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    console.print(f"[dim]Gazetteer: {len(resolver._names)} entities[/dim]\n")
    for m in mentions:
        cands = resolver.resolve(m)
        if cands:
            shown = "  |  ".join(f"{n} ({s:.0f})" for n, s in cands)
            console.print(f"[bold]{m}[/bold] → {shown}")
        else:
            console.print(f"[bold]{m}[/bold] → [yellow]no confident match[/yellow]")


if __name__ == "__main__":
    main()