"""
data_pipeline/wiki_scraper.py

Elden Ring wiki scraper — breadcrumb-recursive crawl.

Strategy:
  Start from 4 top-level entry points that mirror the wiki nav:
    /World+Information
    /Equipment+&+Magic
    /Character+Information
    /Guides+&+Walkthroughs

  From each entry point, recursively follow links whose destination page
  breadcrumb EXTENDS the current page's breadcrumb. This naturally:
    - Discovers all DLC locations (they live under World Information > Locations)
    - Stops at category boundaries (Radahn won't be scraped under Locations
      because his breadcrumb is under Creatures & Enemies, not Locations)
    - Requires zero hardcoded URL lists — the wiki structure itself drives discovery

  Breadcrumb contract:
    - Current page breadcrumb:  ["World Information", "Locations", "Caelid"]
    - Candidate link leads to:  ["World Information", "Locations", "Caelid", "Sellia Crystal Tunnel"]
    - "Sellia Crystal Tunnel" breadcrumb STARTS WITH current → follow it ✓
    - Candidate link leads to:  ["World Information", "Creatures & Enemies", "Radahn"]
    - Does NOT start with current → skip ✗

  HTML structure (confirmed by manual inspection):
    - Breadcrumb: <div id="breadcrumbs-container"> ... <a> segments ... </div>
    - Page title:  <h1><a id="page-title">Page Name</a></h1>
                   or <h1 id="page-title">Page Name</h1>

  Category assignment:
    - Derived from which entry point's subtree the page falls under
    - Refined by breadcrumb segment keywords for cross-cutting pages

Output: data_pipeline/data/pages.jsonl  (one JSON object per line, resumable)

Usage:
    uv run python data_pipeline/wiki_scraper.py --dry-run
    uv run python data_pipeline/wiki_scraper.py --limit 20
    uv run python data_pipeline/wiki_scraper.py
    uv run python data_pipeline/wiki_scraper.py --entry /World+Information --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL    = "https://eldenring.wiki.fextralife.com"
RATE_LIMIT  = float(os.getenv("WIKI_RATE_LIMIT_SECONDS", "1.2"))
CACHE_DIR   = Path("data_pipeline/data/cache")
OUTPUT_FILE = Path("data_pipeline/data/pages.jsonl")
USER_AGENT  = (
    "EldenRingRAGBot/1.0 (portfolio project; educational use; "
    "respectful scraping at ~1 req/sec)"
)

# Maximum recursion depth as a safety net against unexpected cycles.
# In practice the breadcrumb rule terminates naturally well before this.
MAX_DEPTH = 8

# ── Entry points ──────────────────────────────────────────────────────────────
# Each entry point is scraped first, then its child pages are followed
# recursively as long as their breadcrumb extends the parent's.

ENTRY_POINTS: list[tuple[str, str]] = [
    ("/World+Information",        "location"),   # Locations, NPCs, Bosses, Lore, Merchants
    ("/Equipment+&+Magic",        "item"),        # Weapons, Armor, Spells, Talismans, etc.
    ("/Character+Information",    "lore"),        # Classes, Stats, Status Effects
    ("/Guides+&+Walkthroughs",    "quest"),       # Walkthrough, Side Quests, Endings, Crafting
]

# ── Category refinement from breadcrumb keywords ──────────────────────────────
# When a page sits in the World Information tree, its sub-section determines
# a more specific category than the parent entry point's default.

BREADCRUMB_CATEGORY_MAP: list[tuple[str, str]] = [
    # Ordered from most specific to least specific
    ("Bosses",              "boss"),
    ("Creatures",           "boss"),
    ("Enemies",             "boss"),
    ("NPCs",                "quest"),
    ("Merchants",           "quest"),
    ("Side Quests",         "quest"),
    ("Quests",              "quest"),
    ("Walkthrough",         "quest"),
    ("Endings",             "quest"),
    ("Crafting",            "quest"),
    ("Locations",           "location"),
    ("Sites of Grace",      "location"),
    ("Legacy Dungeons",     "location"),
    ("Lore",                "lore"),
    ("Weapons",             "item"),
    ("Armor",               "item"),
    ("Shields",             "item"),
    ("Talismans",           "item"),
    ("Sorceries",           "item"),
    ("Incantations",        "item"),
    ("Spirit Ashes",        "item"),
    ("Ashes of War",        "item"),
    ("Skills",              "item"),
    ("Upgrades",            "item"),
    ("Key Items",           "item"),
    ("Consumables",         "item"),
    ("Stats",               "lore"),
    ("Status Effects",      "lore"),
    ("Classes",             "lore"),
    ("Character",           "lore"),
    ("General Information", "lore"),
]

# ── Pages to never scrape ─────────────────────────────────────────────────────

SKIP_PATHS: frozenset[str] = frozenset({
    "/Interactive+Map",
    "/Maps",
    "/Elden+Ring+Wiki",
    "/Patch+Notes",
    "/Controls",
    "/Multiplayer+Coop+and+Online",
    "/Online+Information",
    "/PvP",
    "/PvE+Builds",
    "/PvP+Builds",
    "/Builds",
    "/Player+Trade",
    "/Build+Calculator",
    "/Summon+Range+Calculator",
    "/New+Game+Plus",
    "/Nightreign",
    "/Elden+Ring+Movie",
    "/Elden+Ring+Tarnished+Edition",
    "/Multiplayer+Items",
    "/Gestures",
    "/Rebirth",
    "/todo",
    "/Trophy+&+Achievement+Guide",
    "/Wiki+Shop",
    "/Covenants",
})

SKIP_URL_FRAGMENTS: tuple[str, ...] = (
    "action=",
    "fextralife.com/Shop",
    "fextralife.com/forums",
    "fextralife.com/news",
    "fextralife.com/blog",
    "wiki.fextralife.com",   # cross-wiki links
    "?",
    "#",
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class WikiPage:
    url:               str
    title:             str
    category:          str
    breadcrumb:        list[str]   # e.g. ["World Information", "Locations", "Caelid"]
    body_text:         str
    infobox:           dict[str, str]
    dialogue:          list[str]
    item_descriptions: list[str]
    image_url:         Optional[str]
    internal_links:    list[str]
    scraped_at:        float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# ── HTTP client ───────────────────────────────────────────────────────────────

def build_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={
            "User-Agent":      USER_AGENT,
            "Accept":          "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
        follow_redirects=True,
        timeout=20.0,
    )


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def _fetch(client: httpx.AsyncClient, url: str) -> str:
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.text


# ── Cache ─────────────────────────────────────────────────────────────────────

def _cache_path(url: str) -> Path:
    return CACHE_DIR / f"{hashlib.md5(url.encode()).hexdigest()}.html"


def _read_cache(url: str) -> Optional[str]:
    p = _cache_path(url)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def _write_cache(url: str, html: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(url).write_text(html, encoding="utf-8")


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    sem: asyncio.Semaphore,
    last_t: list[float],
) -> str:
    """Fetch URL with caching and rate limiting."""
    cached = _read_cache(url)
    if cached:
        return cached
    async with sem:
        elapsed = time.monotonic() - last_t[0]
        if elapsed < RATE_LIMIT:
            await asyncio.sleep(RATE_LIMIT - elapsed)
        html = await _fetch(client, url)
        last_t[0] = time.monotonic()
    _write_cache(url, html)
    return html


# ── URL helpers ───────────────────────────────────────────────────────────────

def to_absolute(href: str) -> Optional[str]:
    """Convert any href to a clean absolute URL, or None if invalid."""
    if not href:
        return None
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = BASE_URL + href
    elif not href.startswith("http"):
        return None
    # Strip fragment
    return href.split("#")[0].rstrip("/") or None


def is_wiki_page(url: str) -> bool:
    """True if this is a scrape-worthy Elden Ring wiki content URL."""
    if not url:
        return False
    parsed = urlparse(url)
    # Must be on the Elden Ring fextralife subdomain
    if "eldenring.wiki.fextralife.com" not in parsed.netloc:
        return False
    path = parsed.path.rstrip("/")
    if not path or path == "/Elden+Ring+Wiki":
        return False
    if path in SKIP_PATHS:
        return False
    if any(frag in url for frag in SKIP_URL_FRAGMENTS):
        return False
    return True


# ── Breadcrumb parsing ────────────────────────────────────────────────────────

def extract_breadcrumb(soup: BeautifulSoup) -> list[str]:
    """
    Extract the breadcrumb trail from the confirmed selector:
        <div id="breadcrumbs-container"> ... <a>Segment</a> ... </div>

    Returns a list of segment strings, e.g.:
        ["World Information", "Locations", "Caelid"]

    Returns [] if no breadcrumb is found (entry-point hub pages often have none).
    """
    container = soup.find(id="breadcrumbs-container")
    if not container:
        return []

    segments: list[str] = []
    for a in container.find_all("a"):
        text = a.get_text(strip=True)
        if text and text not in ("", "Elden Ring Wiki"):
            segments.append(text)

    # Also include any plain text segments (some breadcrumbs use spans, not just <a>)
    if not segments:
        for el in container.find_all(["a", "span", "li"]):
            text = el.get_text(strip=True)
            if text and text not in ("", "Elden Ring Wiki", ">", "/", "»"):
                segments.append(text)

    return segments


def breadcrumb_extends(parent: list[str], child: list[str]) -> bool:
    """
    Return True if child breadcrumb is a direct extension of parent breadcrumb.

    Examples:
        parent = ["World Information", "Locations", "Caelid"]
        child  = ["World Information", "Locations", "Caelid", "Sellia Crystal Tunnel"] → True
        child  = ["World Information", "Locations", "Limgrave"]                        → False
        child  = ["World Information", "Creatures & Enemies", "Radahn"]                → False

    Special case: if parent is [] (entry-point hub with no breadcrumb),
    we accept any child — the hub links are the entry into the tree.
    """
    if not parent:
        return True
    if len(child) != len(parent) + 1:
        return False
    return child[:len(parent)] == parent


# ── Category inference from breadcrumb ───────────────────────────────────────

def infer_category(breadcrumb: list[str], default: str) -> str:
    """
    Walk the breadcrumb from most-specific to least-specific segment
    and return the first category match from BREADCRUMB_CATEGORY_MAP.
    Falls back to `default` (the entry point's base category).
    """
    for segment in reversed(breadcrumb):
        for keyword, category in BREADCRUMB_CATEGORY_MAP:
            if keyword.lower() in segment.lower():
                return category
    return default


# ── HTML parsing — noise stripping ───────────────────────────────────────────

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
    "div#breadcrumbs-container",  # strip breadcrumb from body text
]


def strip_noise(soup: BeautifulSoup) -> None:
    for sel in NOISE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    for tag in ("script", "style", "iframe", "noscript"):
        for el in soup.find_all(tag):
            el.decompose()


# ── Text normalisation ────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Page title ────────────────────────────────────────────────────────────────

def extract_title(soup: BeautifulSoup) -> str:
    # Confirmed selector: <a id="page-title"> inside <h1>
    pt = soup.find(id="page-title")
    if pt:
        return pt.get_text(strip=True)
    h1 = soup.find("h1")
    if h1:
        raw = h1.get_text(strip=True)
        return raw.split("|")[0].strip()
    t = soup.find("title")
    if t:
        return t.get_text(strip=True).split("|")[0].strip()
    return "Unknown"


# ── Image ─────────────────────────────────────────────────────────────────────

def extract_image_url(soup: BeautifulSoup) -> Optional[str]:
    for sel in [
        "div.infobox img",
        "div#wiki-content-block img",
        "table.wiki_table img",
    ]:
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

def _is_nav_list(text: str) -> bool:
    """True if text is a ♦-separated navigation link list."""
    if "♦" not in text or text.count("♦") < 2:
        return False
    segs = [s.strip() for s in text.split("♦") if s.strip()]
    if len(segs) < 2:
        return False
    return sum(1 for s in segs if s and s[0].isupper()) / len(segs) > 0.7


def extract_infobox(soup: BeautifulSoup) -> dict[str, str]:
    infobox: dict[str, str] = {}
    for table in soup.select("table.wiki_table, table.infobox")[:2]:
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) == 2:
                k = normalise(cells[0].get_text(strip=True))
                v = normalise(cells[1].get_text(strip=True))
                if k and v and len(k) < 80 and not _is_nav_list(k) and not _is_nav_list(v):
                    infobox[k] = v
            elif len(cells) == 1:
                h = normalise(cells[0].get_text(strip=True))
                if h and not _is_nav_list(h):
                    infobox["_section"] = h
    return infobox


# ── Dialogue ──────────────────────────────────────────────────────────────────

_DIALOGUE_RE = re.compile(r"dialogue|speech|quotes|voice", re.I)


def extract_dialogue(soup: BeautifulSoup) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        t = normalise(text)
        if len(t) >= 20 and t not in seen:
            seen.add(t)
            lines.append(t)

    # Blockquotes inside tables = NPC speech (most reliable signal)
    for table in soup.find_all("table"):
        for bq in table.find_all("blockquote"):
            _add(bq.get_text(separator=" ", strip=True))

    # Blockquotes / <em> after a "Dialogue" heading
    for heading in soup.find_all(["h2", "h3", "h4"]):
        if not _DIALOGUE_RE.search(heading.get_text()):
            continue
        sibling = heading.find_next_sibling()
        while sibling and isinstance(sibling, Tag):
            if sibling.name == "blockquote":
                _add(sibling.get_text(separator=" ", strip=True))
            elif sibling.name in ("p", "ul", "ol"):
                for em in sibling.find_all(["em", "i"]):
                    _add(em.get_text(separator=" ", strip=True))
            elif sibling.name in ("h2", "h3", "h4"):
                break
            sibling = sibling.find_next_sibling()

    # <em>/<i> inside table cells that look like spoken sentences
    for cell in soup.select("td em, td i"):
        text = cell.get_text(separator=" ", strip=True)
        if len(text) >= 30 and re.search(r'[.!?"\']$', text):
            _add(text)

    return lines


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
                    key=len, default=""
                )
                if len(longest) >= _DESC_MIN:
                    _add(longest)

    return descs


# ── Body text ─────────────────────────────────────────────────────────────────

def _is_nav_diamond(text: str) -> bool:
    if "♦" not in text or text.count("♦") < 2:
        return False
    segs = [s.strip() for s in text.split("♦") if s.strip()]
    return (
        len(segs) >= 2
        and sum(1 for s in segs if s and s[0].isupper()) / len(segs) > 0.7
    )


def extract_body_text(soup: BeautifulSoup) -> str:
    content = (
        soup.select_one("div#wiki-content-block, div.wiki-content, article, div[role='main']")
        or soup.find("body")
        or soup
    )
    lines: list[str] = []
    for el in content.find_all(["p", "li", "h2", "h3", "h4", "td"]):
        text = el.get_text(separator=" ", strip=True)
        if len(text) < 20 or _is_nav_diamond(text):
            continue
        lines.append(text)
    return normalise("\n".join(lines))


# ── Internal links ────────────────────────────────────────────────────────────

def extract_content_links(soup: BeautifulSoup) -> list[str]:
    """
    Extract all wiki content links from the page CONTENT area only.
    Strips the nav first so nav links don't pollute the link set.
    """
    # Remove global nav before harvesting links
    for nav in soup.select("nav, header, div#header, div.wiki-nav"):
        nav.decompose()

    seen: set[str] = set()
    links: list[str] = []
    content = (
        soup.select_one("div#wiki-content-block, div.wiki-content, article, main")
        or soup
    )
    for a in content.find_all("a", href=True):
        url = to_absolute(a["href"])
        if url and is_wiki_page(url) and url not in seen:
            seen.add(url)
            links.append(url)
    return links


# ── Full page parser ──────────────────────────────────────────────────────────

def parse_page(html: str, url: str, category: str) -> WikiPage:
    soup = BeautifulSoup(html, "lxml")

    # Extract breadcrumb BEFORE stripping noise (breadcrumb container is in noise list)
    breadcrumb = extract_breadcrumb(soup)

    # Refine category from breadcrumb
    if breadcrumb:
        category = infer_category(breadcrumb, category)

    strip_noise(soup)

    return WikiPage(
        url=url,
        title=extract_title(soup),
        category=category,
        breadcrumb=breadcrumb,
        body_text=extract_body_text(soup),
        infobox=extract_infobox(soup),
        dialogue=extract_dialogue(soup),
        item_descriptions=extract_item_descriptions(soup),
        image_url=extract_image_url(soup),
        internal_links=extract_content_links(soup),
    )


# ── Breadcrumb-recursive crawler ──────────────────────────────────────────────

async def crawl_recursive(
    client:       httpx.AsyncClient,
    sem:          asyncio.Semaphore,
    last_t:       list[float],
    url:          str,
    parent_crumb: list[str],
    base_category: str,
    discovered:   dict[str, str],   # url → category, shared across all calls
    depth:        int = 0,
) -> None:
    """
    Fetch `url`, record it in `discovered`, then recursively follow every
    content link whose breadcrumb extends the current page's breadcrumb.

    `parent_crumb` is the breadcrumb of the PAGE WE CAME FROM.
    We fetch the current page, parse its breadcrumb, check it extends
    the parent's, then use the current breadcrumb as the new parent_crumb
    for its children.
    """
    if depth > MAX_DEPTH:
        return
    if url in discovered:
        return

    try:
        html = await fetch(client, url, sem, last_t)
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            console.print(f"  [yellow]HTTP {e.response.status_code}[/yellow] {url}")
        return
    except Exception as e:
        console.print(f"  [red]Error[/red] fetching {url}: {e}")
        return

    # Parse breadcrumb from the fetched page
    soup_for_crumb = BeautifulSoup(html, "lxml")
    current_crumb = extract_breadcrumb(soup_for_crumb)

    # Check the breadcrumb contract:
    # The current page's breadcrumb must extend the parent's.
    # Exception: depth=0 means this IS the entry point — always accept.
    if depth > 0 and not breadcrumb_extends(parent_crumb, current_crumb):
        return

    # Record this page
    category = infer_category(current_crumb, base_category)
    discovered[url] = category

    # Collect all content links from the page
    content_links = extract_content_links(BeautifulSoup(html, "lxml"))

    # Recurse into each link
    for child_url in content_links:
        if child_url not in discovered:
            await crawl_recursive(
                client=client,
                sem=sem,
                last_t=last_t,
                url=child_url,
                parent_crumb=current_crumb,
                base_category=base_category,
                discovered=discovered,
                depth=depth + 1,
            )


# ── Discovery orchestrator ────────────────────────────────────────────────────

async def discover_urls(
    client: httpx.AsyncClient,
    sem:    asyncio.Semaphore,
    last_t: list[float],
    entry:  Optional[str] = None,   # restrict to one entry point (for --entry flag)
) -> dict[str, str]:
    """
    Run breadcrumb-recursive crawl from all entry points.
    Returns {url: category_label}.
    """
    discovered: dict[str, str] = {}

    entries = ENTRY_POINTS
    if entry:
        entries = [(p, c) for p, c in ENTRY_POINTS if p == entry]
        if not entries:
            console.print(f"[red]Unknown entry point: {entry}[/red]")
            console.print(f"Valid options: {[p for p,_ in ENTRY_POINTS]}")
            return {}

    for path, base_category in entries:
        url = BASE_URL + path
        console.print(
            f"\n[bold cyan]Crawling entry point:[/bold cyan] {path} "
            f"(base category: {base_category})"
        )
        before = len(discovered)

        await crawl_recursive(
            client=client,
            sem=sem,
            last_t=last_t,
            url=url,
            parent_crumb=[],      # entry points have no parent
            base_category=base_category,
            discovered=discovered,
            depth=0,
        )

        added = len(discovered) - before
        console.print(f"  → {added} URLs discovered under {path}")

    # Summary
    console.print(f"\n[bold green]Discovery complete:[/bold green] {len(discovered)} total URLs\n")
    by_cat: dict[str, int] = defaultdict(int)
    for cat in discovered.values():
        by_cat[cat] += 1
    for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        console.print(f"  {cat:12s}  {count:5d} URLs")

    return discovered


# ── Main scraper ──────────────────────────────────────────────────────────────

async def scrape(
    limit:    Optional[int] = None,
    dry_run:  bool = False,
    entry:    Optional[str] = None,
) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    sem    = asyncio.Semaphore(1)
    last_t = [0.0]

    async with build_client() as client:
        url_map = await discover_urls(client, sem, last_t, entry=entry)

        urls = list(url_map.items())
        if limit:
            urls = urls[:limit]

        if dry_run:
            console.print(
                f"\n[yellow]Dry run — would scrape {len(urls)} pages.[/yellow]\n"
            )
            by_cat: dict[str, list[str]] = defaultdict(list)
            for u, c in urls:
                by_cat[c].append(u)
            for cat, cat_urls in sorted(by_cat.items()):
                console.print(f"  [bold]{cat}[/bold] ({len(cat_urls)} pages):")
                for u in cat_urls[:5]:
                    console.print(f"    {u}")
                if len(cat_urls) > 5:
                    console.print(f"    ... and {len(cat_urls)-5} more")
                console.print()
            return

        console.print(f"\n[bold cyan]Scraping {len(urls)} pages...[/bold cyan]")

        # Resume support
        already: set[str] = set()
        if OUTPUT_FILE.exists():
            with OUTPUT_FILE.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        already.add(json.loads(line)["url"])
                    except Exception:
                        pass
            if already:
                console.print(f"  Resuming: {len(already)} pages already done.")

        errors: list[tuple[str, str]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Scraping...", total=len(urls))

            with OUTPUT_FILE.open("a", encoding="utf-8") as out_f:
                for url, category in urls:
                    progress.update(
                        task, advance=1,
                        description=f"[cyan]{url.split('/')[-1][:45]}"
                    )
                    if url in already:
                        continue
                    try:
                        html = await fetch(client, url, sem, last_t)
                        page = parse_page(html, url, category)
                        if len(page.body_text) < 80:
                            continue
                        out_f.write(
                            json.dumps(page.to_dict(), ensure_ascii=False) + "\n"
                        )
                        out_f.flush()
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
    if errors:
        console.print(f"[yellow]{len(errors)} non-404 errors:[/yellow]")
        for url, err in errors[:10]:
            console.print(f"  {url}: {err}")
        if len(errors) > 10:
            console.print(f"  ...and {len(errors)-10} more")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elden Ring wiki scraper — breadcrumb-recursive"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Discover URLs only, print summary, do not scrape",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Scrape at most N pages (useful for testing)",
    )
    parser.add_argument(
        "--entry", type=str, default=None,
        help="Restrict crawl to one entry point, e.g. /World+Information",
    )
    args = parser.parse_args()
    asyncio.run(scrape(limit=args.limit, dry_run=args.dry_run, entry=args.entry))


if __name__ == "__main__":
    main()