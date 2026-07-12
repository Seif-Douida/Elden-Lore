"""
data_pipeline/chunker.py

Phase 3 of the pipeline: CHUNK.

Reads pages.jsonl (from scrape.py) and produces chunks.jsonl — the units that
will be embedded into Qdrant.

Strategy (see design discussion):
  1. Section-aware splitting — body_text carries '## ' heading markers from the
     scraper; we split on those so each chunk stays within one sub-topic.
  2. Sentence-packing within sections — long sections are packed into chunks of
     ~350 tokens (bge-base tokenizer) with 1-sentence overlap so nothing is lost
     across a boundary. bge-base-en-v1.5 has a 512-token window, leaving ample
     room for the context prefix.
  3. Context injection — every chunk is prefixed with its full path
     [World Information / Locations / Caelid · Bosses] so the embedding encodes
     where the text comes from. This is what makes a chunk about Radahn on the
     Caelid page retrievable for "Caelid bosses", "Radahn drops", etc.
  4. Separate chunk types — body / dialogue / item_desc are tagged distinctly so
     a dialogue-specific question can target dialogue, while default retrieval
     still pulls all types together for a full picture.
  5. Entity tagging — a gazetteer built from every page title + internal-link
     name is matched (case-insensitive, token-boundary) against each chunk; the
     hits go into entities[] for exact-match filtering alongside vector search.
  6. Deduplication — identical raw content is emitted once (first seen wins).

Output: data/chunks.jsonl — one chunk object per line, ready for ingest.py.

Usage:
    uv run python data_pipeline/chunker.py
    uv run python data_pipeline/chunker.py --limit 50 --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from common import DATA_DIR, OUTPUT_FILE  # OUTPUT_FILE == pages.jsonl

console = Console()

# ── Configuration ─────────────────────────────────────────────────────────────

PAGES_FILE  = OUTPUT_FILE                    # data/pages.jsonl
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

EMBED_MODEL = "BAAI/bge-base-en-v1.5"        # sets tokenizer + Qdrant dim (768)

# bge-base window is 512 tokens. We target ~350 tokens of CONTENT per chunk,
# leaving headroom for the context prefix + special tokens.
MAX_CONTENT_TOKENS = 350
OVERLAP_SENTENCES  = 1                        # carry last sentence into next chunk

# Drop chunks whose content is below this — they're template fragments,
# orphaned table cells, or nav labels, not meaningful passages.
MIN_CHUNK_CHARS = 80

# Cap entities stored per chunk. List pages (e.g. "every Ash of War") otherwise
# accumulate 100+ tags, which makes entity-filtering meaningless for that chunk.
MAX_ENTITIES_PER_CHUNK = 30

# Literal template placeholders left in the wiki markup.
_PLACEHOLDER_RE = re.compile(r"\[\s*other\s+\w+\s+go\s+here\s*\]", re.I)

# Section headings often repeat the page title, e.g.
#   "Elden Ring Milady Notes & Tips"  →  "Notes & Tips"
# We strip a leading "Elden Ring <PageTitle>" / "Elden Ring " prefix.
def _clean_heading(heading: str, title: str) -> str:
    h = heading.strip()
    for prefix in (f"Elden Ring {title}", f"{title}", "Elden Ring"):
        if h.lower().startswith(prefix.lower()):
            h = h[len(prefix):].strip(" -–—:·|")
            break
    return h or "Overview"

# Gazetteer hygiene
MIN_ENTITY_LEN = 3                            # drop 1–2 char "entities"
ENTITY_STOPLIST = {
    "map", "the", "and", "lore", "boss", "bosses", "npc", "npcs", "item", "items",
    "weapon", "weapons", "armor", "shield", "shields", "spell", "spells",
    "location", "locations", "enemy", "enemies", "guide", "notes", "note",
}


# ── Tokeniser (bge-base) ──────────────────────────────────────────────────────
# Loaded lazily so --help etc. don't pay the cost.

_tokenizer = None

def _count_tokens(text: str) -> int:
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        console.print(f"[dim]Loading tokenizer: {EMBED_MODEL}[/dim]")
        _tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
    return len(_tokenizer.encode(text, add_special_tokens=False))


# ── Chunk model ───────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:        str
    text:            str          # context prefix + content (this is embedded)
    raw_text:        str          # content only (for display + dedup)
    url:             str
    title:           str
    category:        str
    doc_type:        str          # page | walkthrough
    breadcrumb:      list[str]
    section_heading: str
    chunk_type:      str          # body | dialogue | item_desc
    entities:        list[str]
    image_url:       Optional[str]
    source_type:     str = "wiki"
    facets:          dict = field(default_factory=dict)   # flat, filterable (on every chunk)
    stats:           dict = field(default_factory=dict)   # full stat card (Stats chunk only)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Context prefix ────────────────────────────────────────────────────────────

def build_prefix(breadcrumb: list[str], title: str, section: str) -> str:
    """
    [World Information / Locations / Caelid · Bosses]
    The breadcrumb is the parent path; the page title is appended so the full
    location of the content is encoded. Section appended when meaningful.
    """
    full_path = list(breadcrumb) + [title]
    context = " / ".join(p for p in full_path if p)
    if section and section.lower() not in ("overview", title.lower()):
        context += f" · {section}"
    return f"[{context}]"


# ── Section splitting ─────────────────────────────────────────────────────────

def split_sections(body_text: str) -> list[tuple[str, str]]:
    """
    Split body on '## ' heading markers.
    Returns [(heading, content), ...]. Content before the first heading is
    labelled 'Overview'. Empty sections are dropped.
    """
    sections: list[tuple[str, str]] = []
    heading = "Overview"
    buf: list[str] = []

    for line in body_text.split("\n"):
        if line.startswith("## "):
            if buf:
                content = "\n".join(buf).strip()
                if content:
                    sections.append((heading, content))
            heading = line[3:].strip()
            buf = []
        else:
            buf.append(line)

    if buf:
        content = "\n".join(buf).strip()
        if content:
            sections.append((heading, content))

    return sections


# ── Sentence packing ──────────────────────────────────────────────────────────

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _hard_split_words(sentence: str, max_tokens: int) -> list[str]:
    """Fallback: split an over-long single sentence on word boundaries."""
    words = sentence.split()
    out, cur = [], []
    for w in words:
        cur.append(w)
        if _count_tokens(" ".join(cur)) >= max_tokens:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def pack_sentences(text: str, max_tokens: int, overlap: int) -> list[str]:
    """
    Greedily pack sentences into chunks up to max_tokens, carrying `overlap`
    trailing sentences into the next chunk for continuity.
    """
    sentences: list[str] = []
    for s in _SENT_RE.split(text):
        s = s.strip()
        if not s:
            continue
        if _count_tokens(s) > max_tokens:
            sentences.extend(_hard_split_words(s, max_tokens))
        else:
            sentences.append(s)

    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0

    for s in sentences:
        st = _count_tokens(s)
        if cur and cur_tok + st > max_tokens:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            cur_tok = sum(_count_tokens(x) for x in cur)
        cur.append(s)
        cur_tok += st

    if cur:
        chunks.append(" ".join(cur))
    return chunks


# ── Entity gazetteer + matcher ────────────────────────────────────────────────

def slug_to_name(url: str) -> str:
    """https://.../Starscourge+Radahn → 'Starscourge Radahn'"""
    slug = url.rstrip("/").split("/")[-1]
    return unquote(slug.replace("+", " ")).strip()


def build_gazetteer(pages: list[dict]) -> list[str]:
    """
    Collect entity names from every page title and every internal-link target.
    Deduplicated, stoplisted, and length-filtered.
    """
    names: set[str] = set()
    for p in pages:
        if p.get("title"):
            names.add(p["title"].strip())
        for link in p.get("internal_links", []):
            name = slug_to_name(link)
            if name:
                names.add(name)

    clean: set[str] = set()
    for n in names:
        if len(n) < MIN_ENTITY_LEN:
            continue
        if n.lower() in ENTITY_STOPLIST:
            continue
        clean.add(n)
    return sorted(clean)


def build_matcher(gazetteer: list[str]):
    """spaCy PhraseMatcher (LOWER attr) over the gazetteer. Blank tokenizer = fast."""
    import spacy
    from spacy.matcher import PhraseMatcher

    nlp = spacy.blank("en")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    # Add in batches to keep memory reasonable
    patterns = [nlp.make_doc(name) for name in gazetteer]
    matcher.add("ER", patterns)

    canon_by_lower = {name.lower(): name for name in gazetteer}
    return nlp, matcher, canon_by_lower


def tag_entities(text: str, nlp, matcher, canon_by_lower) -> list[str]:
    doc = nlp.make_doc(text)
    found: set[str] = set()
    for _mid, start, end in matcher(doc):
        span_text = doc[start:end].text
        found.add(canon_by_lower.get(span_text.lower(), span_text))
    return sorted(found)


# ── Facets & structured stats ─────────────────────────────────────────────────

# Weapon-type breadcrumb leaves carry wiki-side typos/variants that fragment
# enumeration ("Glintstone Staffs" vs "Glinstone Staves"). Canonicalize them.
_WEAPON_TYPE_CANON = {
    "glinstone staves": "Glintstone Staffs",
    "glintstone staves": "Glintstone Staffs",
    "ballistas":         "Ballistae",
}


_DLC_MARKERS = ("shadow of the erdtree", "realm of shadow", "land of shadow")


def _is_dlc(page: dict) -> bool:
    """Heuristic DLC flag: a DLC-specific phrase in the page intro. Items say "brand
    new … in the Shadow of the Erdtree DLC"; bosses/locations instead sit "in the
    Realm/Land of Shadow" — the earlier item-only heuristic missed those. Restricted
    to the intro so base pages that merely reference the DLC later aren't tagged."""
    intro = page.get("body_text", "")[:2000].lower()
    return any(m in intro for m in _DLC_MARKERS)


def _canon_weapon_type(leaf: str) -> str:
    return _WEAPON_TYPE_CANON.get(leaf.lower().strip(), leaf)


def compute_facets(page: dict, stats: dict) -> dict:
    """Flat, filterable facets attached to EVERY chunk of a page."""
    f: dict = {"dlc": _is_dlc(page), "subject": page.get("title", "")}
    bc = page.get("breadcrumb", [])
    if "Weapons" in bc and len(bc) >= 3:
        f["weapon_type"] = _canon_weapon_type(bc[-1])
    for attr in ("str", "dex", "int", "fai", "arc"):
        g = stats.get("scaling", {}).get(attr)
        if g:
            f[f"scaling_{attr}"] = g
    if isinstance(stats.get("weight"), (int, float)):
        f["weight"] = stats["weight"]
    if isinstance(stats.get("fp_cost"), int):
        f["fp_cost"] = stats["fp_cost"]
    if stats.get("weak_to"):
        f["weak_to"] = stats["weak_to"]
    return f


def stats_text(stats: dict) -> str:
    """One readable block for a stat card, so the (otherwise payload-only) numbers
    are embedded + retrievable — answers 'give me the stats of X'. The Scaling row
    especially is dropped from body_text by the scraper's nav filter, so this is
    the only place those grades reach the embedding."""
    def _kv(d: dict) -> str:
        return ", ".join(f"{k.title()} {v}" for k, v in d.items())
    parts: list[str] = []
    if stats.get("attack"):
        parts.append("Attack — " + _kv(stats["attack"]))
    if stats.get("scaling"):
        parts.append("Scaling — " + _kv(stats["scaling"]))
    if stats.get("requires"):
        parts.append("Requires — " + _kv(stats["requires"]))
    for k, lab in (("weight", "Weight"), ("fp_cost", "FP Cost"), ("slots", "Slots")):
        if stats.get(k) is not None:
            parts.append(f"{lab} {stats[k]}")
    if stats.get("weak_to"):
        parts.append("Weak to " + ", ".join(stats["weak_to"]))
    if stats.get("strong_vs"):
        parts.append("Strong vs " + ", ".join(stats["strong_vs"]))
    return ". ".join(parts)


# ── Entity denoise ────────────────────────────────────────────────────────────
# The gazetteer matcher tags EVERY page-title/link name in a chunk, so a chunk
# averages ~13 entity tags — mostly noise (breadcrumb nodes, the game name,
# stat/mechanic words). That noise makes entity-filtering imprecise (a chunk that
# merely MENTIONS Rennala matched a Rennala query). Denoise to the real entities,
# and guarantee the page's own subject is present.

_ENTITY_GLOBAL_STOP = {
    "Elden Ring", "World Information", "Equipment & Magic", "Equipment", "Items",
    "Creatures and Enemies", "Guides & Walkthroughs", "Key Items", "Armor",
    "Weapons", "NPCs", "Bosses", "Locations", "Skills", "Magic",
}
_ENTITY_STAT_STOP = {
    "Str", "Dex", "Int", "Fai", "Arc", "Strength", "Dexterity", "Intelligence",
    "Faith", "Arcane", "Fire", "Light", "Holy", "Lightning", "Physical", "Magic",
    "Slash", "Strike", "Pierce", "Standard", "Combat", "Death", "HP", "FP",
    "Info", "Page", "Skill",
}


def _denoise_entities(ents: list[str], breadcrumb: list[str], title: str) -> list[str]:
    crumb = set(breadcrumb)
    out: list[str] = []
    for e in ents:
        if e in _ENTITY_GLOBAL_STOP or e in _ENTITY_STAT_STOP or e in crumb:
            continue
        if e.islower():            # generic mechanic words: "daggers", "skill", "intelligence"
            continue
        out.append(e)
    # The page's subject is always a valid tag for its own chunks (aboutness).
    if title and title not in out and title not in _ENTITY_GLOBAL_STOP:
        out.insert(0, title)
    return out


# ── In-chunk deduplication ────────────────────────────────────────────────────
# Fextralife renders some blocks twice (mobile + desktop, or list items echoed),
# so ~6% of chunks carry a repeated sentence or phrase ("Video Location Video
# Location"; a drops list printed twice). This also corrupted counts (the somber-
# stone "says 6, lists 4" case). Collapse both, word-based (no regex backtracking).

def _dedup_text(text: str) -> str:
    # 1. Drop an adjacent duplicate sentence.
    sents: list[str] = []
    for p in re.split(r"(?<=[.!?])\s+", text):
        if not sents or p.strip().lower() != sents[-1].strip().lower():
            sents.append(p)
    text = " ".join(sents)
    # 2. Collapse an immediately repeated word-run (up to 8 words), e.g.
    #    "A B C A B C" → "A B C". Greedy on the longest run at each position.
    words = text.split()
    out: list[str] = []
    i, n = 0, len(words)
    while i < n:
        run = 0
        for L in range(min(8, (n - i) // 2), 0, -1):
            if words[i:i + L] == words[i + L:i + 2 * L]:
                run = L
                break
        if run:
            out.extend(words[i:i + run])
            i += 2 * run
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


# ── Chunk id ──────────────────────────────────────────────────────────────────

def make_chunk_id(url: str, chunk_type: str, idx: int) -> str:
    return hashlib.md5(f"{url}|{chunk_type}|{idx}".encode()).hexdigest()[:16]


# ── Per-page chunking ─────────────────────────────────────────────────────────

def chunk_page(page: dict, ent_ctx) -> list[Chunk]:
    nlp, matcher, canon = ent_ctx
    url        = page["url"]
    title      = page["title"]
    category   = page["category"]
    doc_type   = page.get("doc_type", "page")
    breadcrumb = page.get("breadcrumb", [])
    image_url  = page.get("image_url")

    stats  = page.get("stats", {}) or {}
    facets = compute_facets(page, stats)

    chunks: list[Chunk] = []
    idx = 0

    def emit(raw: str, section: str, chunk_type: str,
             stats_dict: Optional[dict] = None, allow_short: bool = False) -> None:
        nonlocal idx
        raw = _dedup_text(raw.strip())
        # Drop placeholder fragments and anything too small to be meaningful.
        # `allow_short` exempts the structured Stats chunk (e.g. a boss's short
        # "Weak to Slash" line), which is always meaningful despite its length.
        if not raw or (not allow_short and len(raw) < MIN_CHUNK_CHARS):
            return
        if _PLACEHOLDER_RE.search(raw):
            return
        prefix = build_prefix(breadcrumb, title, section)
        text = f"{prefix}\n{raw}"
        ents = _denoise_entities(tag_entities(text, nlp, matcher, canon), breadcrumb, title)
        if len(ents) > MAX_ENTITIES_PER_CHUNK:
            ents = ents[:MAX_ENTITIES_PER_CHUNK]
        chunks.append(Chunk(
            chunk_id=make_chunk_id(url, chunk_type, idx),
            text=text,
            raw_text=raw,
            url=url,
            title=title,
            category=category,
            doc_type=doc_type,
            breadcrumb=breadcrumb,
            section_heading=section,
            chunk_type=chunk_type,
            entities=ents,
            image_url=image_url,
            facets=facets,               # flat facets on every chunk of the page
            stats=stats_dict or {},      # full stat card only on the Stats chunk
        ))
        idx += 1

    # 0. Structured stats → a readable, retrievable "Stats" chunk. The Scaling row
    #    (and boss weaknesses, spell FP/slots) are dropped from body_text by the
    #    scraper's nav filter, so this is where those numbers reach the embedding.
    st = stats_text(stats)
    if st:
        emit(st, "Stats", "body", stats_dict=stats, allow_short=True)

    # 1. Body — section-aware, then sentence-packed
    for heading, content in split_sections(page.get("body_text", "")):
        heading = _clean_heading(heading, title)
        for piece in pack_sentences(content, MAX_CONTENT_TOKENS, OVERLAP_SENTENCES):
            emit(piece, heading, "body")

    # 2. Dialogue — packed together (preserves conversational flow), own type
    dialogue = page.get("dialogue", [])
    if dialogue:
        joined = "\n".join(dialogue)
        for piece in pack_sentences(joined, MAX_CONTENT_TOKENS, OVERLAP_SENTENCES):
            emit(piece, "Dialogue", "dialogue")

    # 3. Item descriptions — packed, own type
    item_desc = page.get("item_descriptions", [])
    if item_desc:
        joined = "\n".join(item_desc)
        for piece in pack_sentences(joined, MAX_CONTENT_TOKENS, OVERLAP_SENTENCES):
            emit(piece, "Item Description", "item_desc")

    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def run(limit: Optional[int] = None, verbose: bool = False,
        input_file: Optional[str] = None, output_file: Optional[str] = None) -> None:
    # Pages to chunk come from --input (default pages.jsonl). The gazetteer is
    # ALWAYS built from the full pages.jsonl so supplement chunks get complete
    # entity tags even when --input is a smaller supplement file.
    pages_path  = Path(input_file) if input_file else PAGES_FILE
    chunks_path = Path(output_file) if output_file else CHUNKS_FILE

    if not pages_path.exists():
        console.print(f"[red]No pages file at {pages_path}. Run scrape.py first.[/red]")
        return

    pages = [json.loads(l) for l in pages_path.open(encoding="utf-8")]
    if limit:
        pages = pages[:limit]
    console.print(f"[bold cyan]Chunking {len(pages)} pages[/bold cyan]")

    # Build entity gazetteer from the FULL corpus (pages.jsonl), not the --input
    # slice, so entity tagging is complete regardless of what we're chunking now.
    all_pages = [json.loads(l) for l in PAGES_FILE.open(encoding="utf-8")]
    gazetteer = build_gazetteer(all_pages)
    console.print(f"[dim]Entity gazetteer: {len(gazetteer)} names[/dim]")
    ent_ctx = build_matcher(gazetteer)

    seen_hashes: set[str] = set()
    written = 0
    dropped_dupes = 0
    by_type: dict[str, int] = {}

    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress, chunks_path.open("w", encoding="utf-8") as out_f:
        task = progress.add_task("Chunking...", total=len(pages))
        for page in pages:
            progress.update(task, advance=1, description=f"[cyan]{page['title'][:40]}")
            for ch in chunk_page(page, ent_ctx):
                # Dedup PER PAGE (url + text), not globally: some distinct pages
                # legitimately share content (a boss's phase-variant page, numbered
                # item variants). A global raw_text dedup orphaned the second page
                # entirely — losing its chunks AND its subject/facets (e.g. Messmer
                # the Impaler ceded all 55 chunks to Base Serpent Messmer). Keying on
                # the url keeps each page complete while still dropping intra-page
                # repeats (the HTML block-duplication).
                h = hashlib.md5((ch.url + "\n" + ch.raw_text).encode()).hexdigest()
                if h in seen_hashes:
                    dropped_dupes += 1
                    continue
                seen_hashes.add(h)
                out_f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")
                written += 1
                by_type[ch.chunk_type] = by_type.get(ch.chunk_type, 0) + 1
                if verbose:
                    console.print(
                        f"  [green][{ch.chunk_type}][/green] {ch.section_heading[:30]:30s} "
                        f"[dim]{len(ch.entities)} ents[/dim]  {ch.raw_text[:60]}"
                    )

    console.print(f"\n[bold green]Done![/bold green] {written} chunks → {chunks_path}")
    console.print(f"  by type: {by_type}")
    console.print(f"  deduped: {dropped_dupes} duplicate chunks dropped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Elden Ring wiki — chunking phase")
    parser.add_argument("--limit", type=int, default=None,
                        help="Chunk only the first N pages (testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each chunk as it is produced")
    parser.add_argument("--input", type=str, default=None,
                        help="pages jsonl to chunk (default data/pages.jsonl)")
    parser.add_argument("--output", type=str, default=None,
                        help="chunks jsonl to write (default data/chunks.jsonl)")
    args = parser.parse_args()
    run(limit=args.limit, verbose=args.verbose,
        input_file=args.input, output_file=args.output)


if __name__ == "__main__":
    main()