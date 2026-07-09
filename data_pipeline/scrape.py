"""
data_pipeline/scrape.py

Phase 2 of the pipeline: SCRAPE.

Reads the manifest produced by discover.py (discovered_urls.jsonl), parses
each page, and writes structured records to pages.jsonl.

Because discovery already fetched every page into the HTML cache, scraping is
normally cache-only and runs in seconds — so you can re-run it freely after
tweaking any extractor below (dialogue, item descriptions, body cleaning…)
without re-crawling the wiki. If a page somehow isn't cached, it is fetched
on demand (rate-limited).

Each output record (see common.WikiPage):
    url, title, category, breadcrumb,
    body_text, infobox, dialogue, item_descriptions,
    image_url, internal_links, scraped_at

Usage:
    uv run python data_pipeline/scrape.py
    uv run python data_pipeline/scrape.py --verbose
    uv run python data_pipeline/scrape.py --limit 50
    uv run python data_pipeline/scrape.py --fresh        # ignore existing pages.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from common import (
    BASE_URL,
    DISCOVERED_FILE, OUTPUT_FILE,
    WikiPage, console, build_client, fetch, is_cached,
    to_absolute, is_wiki_page,
    extract_breadcrumb, extract_title, infer_category,
)


# ── Text normalisation ────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Noise stripping ───────────────────────────────────────────────────────────

NOISE_SELECTORS: list[str] = [
    "div#wiki-commentslist", "div.wiki-commentslist",
    "div.comments", "div#comments",
    "div#wiki-content-block > div.col-sm-3",
    "nav", "div.wiki-nav", "div#header", "header",
    "div#footer", "footer",
    "div.ads", "div.adsbygoogle", "div[class*='ad-']", "div[class*='social']",
    "span.wiki-edit", "div.wiki-page-edit",
    "div.wiki-page-discussion",
    "div.wiki-rating", "div[class*='rating']",
    "div#wiki-content-block div.toc",
    "div#breadcrumbs-container",
]


def strip_noise(soup: BeautifulSoup) -> None:
    for sel in NOISE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    for tag in ("script", "style", "iframe", "noscript"):
        for el in soup.find_all(tag):
            el.decompose()


# ── Diamond nav-list detection ────────────────────────────────────────────────

def _is_nav_list(text: str) -> bool:
    """True if text is a ♦-separated navigation link list (not real content)."""
    if "♦" not in text or text.count("♦") < 2:
        return False
    segs = [s.strip() for s in text.split("♦") if s.strip()]
    if len(segs) < 2:
        return False
    return sum(1 for s in segs if s and s[0].isupper()) / len(segs) > 0.7


# ── Image ─────────────────────────────────────────────────────────────────────

def extract_image_url(soup: BeautifulSoup) -> Optional[str]:
    for sel in ["div.infobox img", "div#wiki-content-block img", "table.wiki_table img"]:
        for img in soup.select(sel):
            src = img.get("src", "")
            if not src:
                continue
            if any(x in src.lower() for x in ("icon", "logo", "banner", "arrow")):
                continue
            try:
                if int(img.get("width", 200)) < 80:
                    continue
            except (ValueError, TypeError):
                pass
            if src.startswith("//"):
                return "https:" + src
            if src.startswith("/"):
                return BASE_URL + src
            return src
    return None


# ── Infobox ───────────────────────────────────────────────────────────────────

def _is_mashed_stats(key: str, val: str) -> bool:
    """
    True for stat-table rows that got flattened into unreadable blobs, e.g.
        key="AttackPhy116Mag0Fire0Ligt0Holy0Crit100"  val="GuardPhy 63Mag31..."
        key="ScalingStrDDexC"                          val="RequiresStr18Dex17"
        key="Wgt.6.5"                                  val="Passive-"
    These come from weapon/boss stat grids that have no cell spacing. The same
    numbers appear cleanly in body_text, so we drop the mangled infobox copy.

    We must NOT drop short legitimate values like "FP9" (an Ash of War's FP cost)
    or "FP8 ( - 12)". The distinguishing feature of a mashed grid is *multiple*
    run-together stat groups, so we require 2+ digit-runs (or a Scaling/Requires
    CamelCase key) rather than firing on any single letter-digit token.
    """
    for s in (key, val):
        if not s:
            continue
        # 2+ separate digit-runs crammed into a (near-)spaceless token → mashed
        if len(re.findall(r"\d+", s)) >= 2 and s.count(" ") <= 1:
            return True
    # A "Scaling…"/"Requires…" key with run-together CamelCase stat codes
    if re.search(r"(Scaling|Requires)[A-Z].*[A-Z]", key) and " " not in key:
        return True
    return False


def extract_infobox(soup: BeautifulSoup) -> dict[str, str]:
    infobox: dict[str, str] = {}
    for table in soup.select("table.wiki_table, table.infobox")[:2]:
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) == 2:
                k = normalise(cells[0].get_text(strip=True))
                v = normalise(cells[1].get_text(strip=True))
                if not k or not v or len(k) >= 80:
                    continue
                if _is_nav_list(k) or _is_nav_list(v):
                    continue
                if _is_mashed_stats(k, v):
                    continue            # garbled stat grid — body has it cleanly
                infobox[k] = v
            # Single-cell rows are section headings (page title etc.) — skip them.
            # We no longer store a "_section" key; it was pure noise.
    return infobox


# ── Structured stats (weapon / spell / boss / armor stat cards) ───────────────
# Fextralife renders stat cards as: a label (text and/or an <img title>) followed
# by <div class="lineleft"> holding values as <a>Name</a> value pairs — plus plain
# <td>Label Value</td> cells (FP Cost, Slots Used, Wgt.). The generic body-text
# extractor drops the SHORT cells (2-attribute scaling, "Weak to X", "FP Cost 7")
# via its len<20 nav filter, so we parse them structurally here. First clean
# occurrence of each label wins.

_STAT_GRADES = set("SABCDE")
_STAT_ATTR = {
    "str": "str", "strength": "str", "dex": "dex", "dexterity": "dex",
    "int": "int", "intelligence": "int", "fai": "fai", "faith": "fai",
    "arc": "arc", "arcane": "arc",
}
_STAT_LABELS = {
    "attack": "attack", "guard": "guard", "scaling": "scaling",
    "requires": "requires", "weak to": "weak_to", "strong vs": "strong_vs",
    "resistant": "resistant", "immune to": "immune", "effect": "effect",
}


def _stat_label(div: Tag) -> Optional[str]:
    """The stat label preceding a div.lineleft — the LAST known label keyword in the
    text before it (handles both 'Weak to<br><div>' and 'Effect…Requires<div>')."""
    parts: list[str] = []
    for node in div.parent.children:
        if node is div:
            break
        if isinstance(node, NavigableString):
            parts.append(str(node))
        elif isinstance(node, Tag) and node.name != "img":
            parts.append(node.get_text(" "))
    text = re.sub(r"\s+", " ", "".join(parts)).lower()
    best, pos = None, -1
    for lab in _STAT_LABELS:
        i = text.rfind(lab)
        if i > pos:
            pos, best = i, lab
    return best


def _lineleft_pairs(div: Tag):
    """(pairs, text): a div.lineleft yields <a>Name</a> value pairs, or plain prose
    (e.g. an Effect blurb) when it holds no anchors."""
    anchors = div.find_all("a")
    if not anchors:
        return None, div.get_text(" ", strip=True).replace("\xa0", " ").strip()
    pairs = []
    for a in anchors:
        name = a.get_text(" ", strip=True)
        val, sib = "", a.next_sibling
        while sib and not (isinstance(sib, Tag) and sib.name in ("a", "br")):
            if isinstance(sib, NavigableString):
                val += str(sib)
            sib = sib.next_sibling
        pairs.append((name, val.replace("\xa0", " ").strip()))
    return pairs, None


def extract_stats(soup: BeautifulSoup) -> dict:
    """Structured stat card → e.g.
    {"scaling": {"str":"E","dex":"D"}, "requires": {"str":16,"dex":48},
     "attack": {"phy":117}, "weak_to": ["Slash"], "weight": 7.0, "fp_cost": 7}."""
    stats: dict = {}
    for div in soup.select("table.wiki_table div.lineleft, table.infobox div.lineleft"):
        key = _STAT_LABELS.get(_stat_label(div) or "")
        if not key or key in stats:
            continue
        pairs, text = _lineleft_pairs(div)
        if key in ("scaling", "requires", "attack", "guard"):
            d: dict = {}
            for name, val in (pairs or []):
                nm = _STAT_ATTR.get(name.lower(), name.lower())
                if key == "scaling":
                    if val[:1].upper() in _STAT_GRADES:
                        d[nm] = val[:1].upper()
                else:
                    m = re.search(r"-?\d+", val)
                    if m:
                        d[nm] = int(m.group())
            if d:
                stats[key] = d
        elif key in ("weak_to", "strong_vs", "resistant", "immune"):
            names = [n for n, _ in pairs] if pairs else ([text] if text else [])
            if names:
                stats[key] = names
        elif key == "effect" and (text or pairs):
            stats["effect"] = text or " ".join(n for n, _ in pairs)

    # Plain stat cells the div.lineleft parser doesn't cover.
    for td in soup.select("table.wiki_table td, table.infobox td"):
        t = td.get_text(" ", strip=True).replace("\xa0", " ")
        for pat, k, cast in (
            (r"FP Cost\s*([\d.]+)", "fp_cost", int),
            (r"Slots?\s*Used\s*(\d+)", "slots", int),
            (r"Wgt\.?\s*([\d.]+)", "weight", float),
            (r"Weight\s*([\d.]+)", "weight", float),
        ):
            m = re.match(pat, t)
            if m and k not in stats:
                try:
                    stats[k] = cast(m.group(1))
                except ValueError:
                    pass
    return stats


# ── Dialogue ──────────────────────────────────────────────────────────────────

_DIALOGUE_RE = re.compile(r"dialogue|speech|quotes|voice", re.I)


def extract_dialogue(soup: BeautifulSoup) -> list[str]:
    """
    Extract genuine NPC dialogue. Two reliable signals only:
      1. <blockquote> inside a table  → NPC speech blocks on Fextralife
      2. content under a 'Dialogue' / 'Speech' / 'Quotes' heading

    We deliberately do NOT scan generic <td><em> cells: on location pages
    those are italicised description captions ("A dilapidated church found
    east of Sellia...") which are NOT dialogue and were polluting the field.
    """
    lines: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        t = normalise(text)
        if len(t) >= 15 and t not in seen:
            seen.add(t)
            lines.append(t)

    # A page counts as dialogue-bearing only if it has a Dialogue/Speech heading.
    # We use that to gate the standalone-blockquote pass, so item flavour text
    # (also a blockquote) isn't mis-captured as dialogue on non-NPC pages.
    has_dialogue_section = any(
        _DIALOGUE_RE.search(h.get_text()) for h in soup.find_all(["h2", "h3", "h4"])
    )

    # 1. Blockquotes inside tables — the most reliable NPC-speech signal.
    for table in soup.find_all("table"):
        for bq in table.find_all("blockquote"):
            _add(bq.get_text(separator=" ", strip=True))

    # 1b. Standalone blockquotes (an NPC's signature line often sits outside any
    #     table) — but only on pages that actually have a dialogue section.
    if has_dialogue_section:
        content = soup.select_one(
            "div#wiki-content-block, div.wiki-content, article"
        ) or soup
        for bq in content.find_all("blockquote"):
            if not bq.find_parent("table"):
                _add(bq.get_text(separator=" ", strip=True))

    # 2. Content under an explicit Dialogue / Speech / Quotes heading. Each
    #    container's <em>/<i> lines are MERGED into one speech (Fextralife renders
    #    a single spoken passage as many italic subtitle fragments — emitting one
    #    entry per <em> shattered multi-line dialogue). We also scan <div>, not
    #    just <p>/<ul>/<ol>, because some NPCs' lines sit in <div> wrappers.
    for heading in soup.find_all(["h2", "h3", "h4"]):
        if not _DIALOGUE_RE.search(heading.get_text()):
            continue
        sibling = heading.find_next_sibling()
        while sibling and isinstance(sibling, Tag):
            if sibling.name in ("h2", "h3", "h4"):
                break
            if sibling.name == "blockquote":
                _add(sibling.get_text(separator=" ", strip=True))
            elif sibling.name in ("p", "div", "ul", "ol"):
                ems = sibling.find_all(["em", "i"])
                if ems:
                    _add(" ".join(em.get_text(separator=" ", strip=True) for em in ems))
            sibling = sibling.find_next_sibling()

    # Fextralife renders the same speech in both a table and a blockquote, so a
    # line often appears twice — once partial, once full. Drop any entry that is a
    # substring of another (keep the longer), preserving first-seen order.
    return [
        t for t in lines
        if not any(t != other and t in other for other in lines)
    ]


# ── Item descriptions ─────────────────────────────────────────────────────────

_DESC_RE  = re.compile(r"description|lore text|flavou?r|in-game", re.I)
_DESC_MIN = 40


def extract_item_descriptions(soup: BeautifulSoup) -> list[str]:
    descs: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        t = normalise(text)
        if len(t) >= _DESC_MIN and t not in seen:
            seen.add(t)
            descs.append(t)

    for el in soup.select("div.codex, span.codex"):
        _add(el.get_text(separator=" ", strip=True))

    # The in-game flavour text sits in a div.lineleft of italic <em> paragraphs —
    # the same container class as the stat boxes, but prose with NO <a> stat
    # anchors. That distinction is what makes it separable (the old codex/heading
    # paths caught almost nothing, ~1% of item pages).
    for div in soup.select("div.lineleft"):
        if div.find(["em", "i"]) and not div.find("a"):
            _add(div.get_text(separator=" ", strip=True))

    for heading in soup.find_all(["h2", "h3", "h4"]):
        if not _DESC_RE.search(heading.get_text()):
            continue
        sibling = heading.find_next_sibling()
        collected = 0
        while sibling and isinstance(sibling, Tag) and collected < 3:
            if sibling.name == "p":
                t = sibling.get_text(separator=" ", strip=True)
                if _DESC_MIN <= len(t) <= 1000:
                    _add(t)
                    collected += 1
            elif sibling.name in ("h2", "h3", "h4"):
                break
            sibling = sibling.find_next_sibling()

    for table in soup.select("table.wiki_table"):
        headers = [th.get_text(strip=True).lower() for th in table.select("th")]
        if not any("description" in h or "effect" in h or "lore" in h for h in headers):
            continue
        for row in table.select("tr"):
            cells = row.find_all("td")
            if cells:
                longest = max(
                    (c.get_text(separator=" ", strip=True) for c in cells),
                    key=len, default="",
                )
                if len(longest) >= _DESC_MIN:
                    _add(longest)

    return descs


# ── Body text ─────────────────────────────────────────────────────────────────

# Inline-noise patterns to remove from body lines.
_MAP_LINK_RE   = re.compile(r"\[\s*(Elden Ring\s+)?Map Link\s*\]", re.I)
_BRACKET_NOISE = re.compile(r"\[\s*\]")  # empty brackets left after removal


def _clean_body_line(text: str) -> str:
    text = _MAP_LINK_RE.sub("", text)
    text = _BRACKET_NOISE.sub("", text)
    return normalise(text)


def extract_body_text(soup: BeautifulSoup) -> str:
    """
    Build the body text while PRESERVING heading structure so the chunker can
    later split pages by section. Headings (h2/h3/h4) are emitted on their own
    line prefixed with a '## ' marker; the chunker splits on these.

    Also removes inline '[ Map Link ]' noise and skips ♦ nav lists.
    """
    content = (
        soup.select_one("div#wiki-content-block, div.wiki-content, article, div[role='main']")
        or soup.find("body")
        or soup
    )

    out: list[str] = []
    seen_headings: set[str] = set()

    for el in content.find_all(["p", "li", "h2", "h3", "h4", "td"]):
        raw = el.get_text(separator=" ", strip=True)
        if not raw:
            continue

        if el.name in ("h2", "h3", "h4"):
            heading = _clean_body_line(raw)
            # Skip empty or duplicate headings (Fextralife repeats the tab labels)
            if not heading or heading in seen_headings:
                continue
            seen_headings.add(heading)
            out.append(f"## {heading}")
            continue

        text = _clean_body_line(raw)
        if len(text) < 20 or _is_nav_list(text):
            continue
        out.append(text)

    return normalise("\n".join(out))


# ── Internal links (for the record; tables kept intact here) ──────────────────

def extract_internal_links(soup: BeautifulSoup) -> list[str]:
    seen: set[str] = set()
    links: list[str] = []
    content = soup.select_one("div#wiki-content-block, div.wiki-content, article") or soup
    for a in content.find_all("a", href=True):
        url = to_absolute(a["href"])
        if url and is_wiki_page(url) and url not in seen:
            seen.add(url)
            links.append(url)
    return links


# ── Full page parser ──────────────────────────────────────────────────────────

# A page is treated as a "walkthrough" (huge, section-structured, chunked
# specially) if its title says so or its body is very large.
_WALKTHROUGH_TITLE_RE = re.compile(r"walkthrough|game progress route", re.I)
_WALKTHROUGH_MIN_LEN  = 50_000


def _detect_doc_type(title: str, breadcrumb: list[str], body_len: int) -> str:
    crumb = " / ".join(breadcrumb).lower()
    if _WALKTHROUGH_TITLE_RE.search(title):
        return "walkthrough"
    if "walkthrough" in crumb and body_len >= _WALKTHROUGH_MIN_LEN:
        return "walkthrough"
    if body_len >= _WALKTHROUGH_MIN_LEN:
        return "walkthrough"
    return "page"


def parse_page(html: str, url: str, category: str) -> WikiPage:
    # Breadcrumb is read before noise stripping (its container is in NOISE_SELECTORS)
    soup = BeautifulSoup(html, "lxml")
    breadcrumb = extract_breadcrumb(soup)

    # The manifest already carries a category, but we re-confirm from the
    # breadcrumb here so re-running scrape picks up any category-map changes.
    if breadcrumb:
        category = infer_category(breadcrumb, category)

    title = extract_title(soup)
    internal_links = extract_internal_links(soup)

    strip_noise(soup)

    body_text = extract_body_text(soup)
    doc_type = _detect_doc_type(title, breadcrumb, len(body_text))

    return WikiPage(
        url=url,
        title=title,
        category=category,
        doc_type=doc_type,
        breadcrumb=breadcrumb,
        body_text=body_text,
        infobox=extract_infobox(soup),
        stats=extract_stats(soup),
        dialogue=extract_dialogue(soup),
        item_descriptions=extract_item_descriptions(soup),
        image_url=extract_image_url(soup),
        internal_links=internal_links,
    )


# ── Manifest loader ───────────────────────────────────────────────────────────

def load_manifest() -> list[tuple[str, str]]:
    if not DISCOVERED_FILE.exists():
        return []
    out: list[tuple[str, str]] = []
    with DISCOVERED_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                out.append((obj["url"], obj["category"]))
            except Exception:
                pass
    return out


# ── Main scrape ───────────────────────────────────────────────────────────────

async def scrape(
    limit:   Optional[int] = None,
    verbose: bool = False,
    fresh:   bool = False,
) -> None:
    manifest = load_manifest()
    if not manifest:
        console.print(
            f"[red]No manifest at {DISCOVERED_FILE}.[/red] "
            f"Run discover.py first."
        )
        return

    if limit:
        manifest = manifest[:limit]

    console.print(
        f"[bold cyan]Scraping {len(manifest)} pages[/bold cyan] "
        f"(from manifest {DISCOVERED_FILE})"
    )

    # Resume support, unless --fresh
    already: set[str] = set()
    if fresh and OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
    elif OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(encoding="utf-8") as f:
            for line in f:
                try:
                    already.add(json.loads(line)["url"])
                except Exception:
                    pass
        if already:
            console.print(f"  Resuming: {len(already)} pages already scraped.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sem    = asyncio.Semaphore(1)
    last_t = [0.0]
    errors: list[tuple[str, str]] = []
    uncached = 0

    async with build_client() as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scraping...", total=len(manifest))
            with OUTPUT_FILE.open("a", encoding="utf-8") as out_f:
                for url, category in manifest:
                    progress.update(task, advance=1,
                                    description=f"[cyan]{url.split('/')[-1][:45]}")
                    if url in already:
                        continue
                    if not is_cached(url):
                        uncached += 1
                    try:
                        html = await fetch(client, url, sem, last_t, verbose=verbose)
                        page = parse_page(html, url, category)
                        if len(page.body_text) < 80:
                            if verbose:
                                console.print(
                                    f"  [yellow][skip — too short][/yellow] "
                                    f"{url.split('/')[-1]}"
                                )
                            continue
                        out_f.write(json.dumps(page.to_dict(), ensure_ascii=False) + "\n")
                        out_f.flush()
                        if verbose:
                            crumb = " / ".join(page.breadcrumb) if page.breadcrumb else "—"
                            console.print(
                                f"  [green][ok][/green] {page.title[:45]}  "
                                f"[dim][{page.category}] {crumb}[/dim]"
                            )
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code != 404:
                            errors.append((url, f"HTTP {e.response.status_code}"))
                    except Exception as e:
                        errors.append((url, str(e)))

    total = 0
    if OUTPUT_FILE.exists():
        with OUTPUT_FILE.open(encoding="utf-8") as f:
            total = sum(1 for _ in f)

    console.print(f"\n[bold green]Done![/bold green] {total} pages → {OUTPUT_FILE}")
    if uncached:
        console.print(f"[dim]{uncached} pages were fetched live (not in cache).[/dim]")
    if errors:
        console.print(f"[yellow]{len(errors)} non-404 errors:[/yellow]")
        for url, err in errors[:10]:
            console.print(f"  {url}: {err}")
        if len(errors) > 10:
            console.print(f"  ...and {len(errors)-10} more")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elden Ring wiki — scrape phase (parse from cache)"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Scrape at most N pages from the manifest")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each page as it is parsed")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore and overwrite any existing pages.jsonl")
    args = parser.parse_args()
    asyncio.run(scrape(limit=args.limit, verbose=args.verbose, fresh=args.fresh))


if __name__ == "__main__":
    main()