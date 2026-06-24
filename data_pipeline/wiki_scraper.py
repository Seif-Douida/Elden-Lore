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
DISCOVERED_FILE = Path("data_pipeline/data/discovered_urls.jsonl")
NO_BREADCRUMB_FILE = Path("data_pipeline/data/no_breadcrumb.txt")
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
    # Test / partial entry points (use via --entry):
    ("/Caelid",                   "location"),    # single region, for testing the DFS
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
    "/Shop",
    "/forums",
    "/news",
    "/blog",
    "?",
    "#",
)

# Domains that are NOT the Elden Ring wiki (other fextralife wikis, etc.)
# Checked against parsed.netloc, not the full URL string
SKIP_DOMAINS: frozenset[str] = frozenset({
    "fextralife.com",                    # root domain (shop, forums, blog)
    "darksouls3.wiki.fextralife.com",
    "darksouls.wiki.fextralife.com",
    "bloodborne.wiki.fextralife.com",
    "sekiro.wiki.fextralife.com",
})


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
    verbose: bool = False,
) -> str:
    """Fetch URL with caching and rate limiting."""
    cached = _read_cache(url)
    if cached:
        if verbose:
            console.print(f"  [dim][cache][/dim] {url}")
        return cached
    async with sem:
        elapsed = time.monotonic() - last_t[0]
        if elapsed < RATE_LIMIT:
            await asyncio.sleep(RATE_LIMIT - elapsed)
        if verbose:
            console.print(f"  [cyan][fetch][/cyan] {url}")
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

    # Must be exactly the Elden Ring wiki subdomain
    if parsed.netloc != "eldenring.wiki.fextralife.com":
        return False

    # Reject other fextralife wikis that might appear as cross-links
    if parsed.netloc in SKIP_DOMAINS:
        return False

    path = parsed.path.rstrip("/")
    if not path or path == "/Elden+Ring+Wiki":
        return False
    if path in SKIP_PATHS:
        return False

    # Reject image/file URLs (e.g. /file/Elden-Ring/radagon...jpg)
    if path.startswith("/file/") or path.lower().endswith(
        (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".pdf")
    ):
        return False

    # Reject query strings, fragments, and specific path substrings
    if parsed.query:
        return False
    if any(frag in url for frag in SKIP_URL_FRAGMENTS):
        return False

    return True


# ── Breadcrumb parsing ────────────────────────────────────────────────────────

def extract_breadcrumb(soup: BeautifulSoup) -> list[str]:
    """
    Extract the breadcrumb trail from:
        <div id="breadcrumbs-container">
          <a href="...">World Information</a>
          <a href="...">Locations</a>
          <!-- hidden editor button: <div id="breadcrumbs-bcontainer">
                 <a href="#" id="btnCreateBreadcrumb">+</a>
               </div> -->
        </div>

    Returns e.g. ["World Information", "Locations", "Caelid"].
    Returns [] if no real breadcrumb segments exist (entry-point hub pages).
    """
    container = soup.find(id="breadcrumbs-container")
    if not container:
        return []

    # The hidden editor sub-div must be excluded — it contains the '+' button
    editor_div = container.find(id="breadcrumbs-bcontainer")
    if editor_div:
        editor_div.decompose()

    segments: list[str] = []
    for a in container.find_all("a"):
        href = a.get("href", "")
        text = a.get_text(strip=True)

        # Skip: empty text, the wiki home link, anchor-only links (#), symbol-only
        if not text:
            continue
        if text in ("Elden Ring Wiki", ">", "/", "»", "+"):
            continue
        if href == "#" or href == "":
            continue
        # Skip single punctuation / symbol segments
        if len(text) <= 1:
            continue

        segments.append(text)

    return segments


def breadcrumb_extends(parent: list[str], child: list[str]) -> bool:
    """
    Return True if the child belongs to the same wiki tree as the parent.

    Rules:
      1. parent is [] (entry point, no breadcrumb) → accept anything
      2. child is []  (page has no breadcrumb set) → accept, but crawl_recursive
                                                      will NOT recurse further
      3. child[:len(parent)] == parent              → child is at the same level
                                                      or deeper in the same tree → accept
      4. anything else                              → different tree → reject

    Real-world examples with parent = ['World Information']:
      child = []                                           → True  (no breadcrumb, accept once)
      child = ['World Information']                        → True  (same level, still our tree)
      child = ['World Information', 'Locations']           → True  (one level deeper)
      child = ['World Information', 'Locations', 'Caelid'] → True  (two levels deeper)
      child = ['Equipment & Magic', 'Weapons']             → False (different entry point)
      child = ['Character Information']                    → False (different entry point)
    """
    if not parent:
        return True
    if not child:
        return True   # no breadcrumb — accept but caller stops recursion
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
        return pt.get_text(strip=True).split("|")[0].strip()
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True).split("|")[0].strip()
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

def extract_content_links(soup: BeautifulSoup, exclude_tables: bool = False) -> list[str]:
    """
    Extract all wiki content links from the page content area only.

    Targets div#wiki-content-block exclusively — this element never contains
    the global nav bar, so we don't need to decompose anything.
    Falls back to the full soup only if the content block is absent,
    in which case we explicitly skip any <nav>/<header> descendants.

    If exclude_tables=True, the wiki_table elements are removed before link
    harvesting. These tables (usually near the bottom of location pages) are
    full alphabetical indexes linking to every other location — following them
    during DFS discovery causes thousands of wasted fetches. We pass
    exclude_tables=True during discovery and leave it False during parsing
    (where the infobox table is still needed for stats extraction).

    The soup passed in is mutated if exclude_tables=True, so callers that need
    the tables intact afterward must pass a fresh soup.
    """
    seen: set[str] = set()
    links: list[str] = []

    if exclude_tables:
        # Remove index/navigation tables before harvesting links.
        # wiki_table = the alphabetical location/item index tables.
        for table in soup.select("table.wiki_table, table.sortable, table.infobox"):
            table.decompose()

    content = soup.select_one("div#wiki-content-block, div.wiki-content, article")

    if content:
        for a in content.find_all("a", href=True):
            url = to_absolute(a["href"])
            if url and is_wiki_page(url) and url not in seen:
                seen.add(url)
                links.append(url)
    else:
        nav_els = set()
        for nav in soup.select("nav, header, div#header, div.wiki-nav, footer"):
            nav_els.update(id(el) for el in nav.find_all("a"))

        for a in soup.find_all("a", href=True):
            if id(a) in nav_els:
                continue
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

def _log_no_breadcrumb(url: str, no_crumb: set[str]) -> None:
    """Record a URL that has no breadcrumb for later review (deduplicated)."""
    if url in no_crumb:
        return
    no_crumb.add(url)
    NO_BREADCRUMB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NO_BREADCRUMB_FILE.open("a", encoding="utf-8") as f:
        f.write(url + "\n")


async def crawl_recursive(
    client:        httpx.AsyncClient,
    sem:           asyncio.Semaphore,
    last_t:        list[float],
    url:           str,
    parent_crumb:  list[str],       # breadcrumb of the page we came from
    base_category: str,
    discovered:    dict[str, str],
    no_crumb:      set[str],         # URLs with no breadcrumb, logged for review
    limit:         Optional[int] = None,
    verbose:       bool = False,
    depth:         int = 0,
) -> None:
    """
    True one-level DFS using the breadcrumb as the descent rule.

    To accept and recurse into `url`, its own breadcrumb `C` must satisfy:
        len(C) == len(parent_crumb) + 1      (exactly one level deeper)
        AND C[:len(parent_crumb)] == parent_crumb   (same path prefix)

    Then `C` becomes the parent_crumb for this page's own children.

    Special cases:
      - depth == 0 → this is the tree root, already validated by the caller.
        We use its breadcrumb directly as parent_crumb for its children.
      - Page has no breadcrumb → log the URL to NO_BREADCRUMB_FILE, do not
        scrape, do not recurse.
    """
    if depth > MAX_DEPTH:
        return
    if depth > 0:
        if url in discovered:
            return
        if limit is not None and len(discovered) >= limit:
            return

    try:
        html = await fetch(client, url, sem, last_t, verbose=(verbose and depth > 0))
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            console.print(f"  [yellow]HTTP {e.response.status_code}[/yellow] {url}")
        return
    except Exception as e:
        console.print(f"  [red]Error[/red] fetching {url}: {e}")
        return

    current_crumb = extract_breadcrumb(BeautifulSoup(html, "lxml"))

    if depth == 0:
        # Tree root — caller already validated it. Record it and use its
        # breadcrumb as the parent for children.
        category = infer_category(current_crumb, base_category)
        discovered[url] = category
        crumb_for_children = current_crumb
    else:
        # No breadcrumb → log and stop (no scrape, no recurse)
        if not current_crumb:
            _log_no_breadcrumb(url, no_crumb)
            if verbose:
                console.print(f"  [dim][no-breadcrumb, logged][/dim] {url}")
            return

        # One-level DFS rule: child must be exactly one segment deeper
        # AND share the full parent prefix.
        if len(current_crumb) != len(parent_crumb) + 1:
            return
        if current_crumb[:len(parent_crumb)] != parent_crumb:
            return

        category = infer_category(current_crumb, base_category)
        discovered[url] = category
        crumb_for_children = current_crumb

        if verbose:
            crumb_str = " / ".join(current_crumb)
            console.print(
                f"  [#{len(discovered):04d}] [{category}] {url.split('/')[-1]}  "
                f"[dim]{crumb_str}[/dim]"
            )
        if limit is not None and len(discovered) >= limit:
            return

    # Recurse into content links — children must extend crumb_for_children by one.
    # exclude_tables=True drops the bottom index tables that link to every
    # other location/item, which would otherwise cause thousands of wasted fetches.
    content_links = extract_content_links(BeautifulSoup(html, "lxml"), exclude_tables=True)
    for child_url in content_links:
        if child_url not in discovered:
            await crawl_recursive(
                client=client,
                sem=sem,
                last_t=last_t,
                url=child_url,
                parent_crumb=crumb_for_children,
                base_category=base_category,
                discovered=discovered,
                no_crumb=no_crumb,
                limit=limit,
                verbose=verbose,
                depth=depth + 1,
            )


# ── Discovery orchestrator ────────────────────────────────────────────────────

async def discover_urls(
    client:  httpx.AsyncClient,
    sem:     asyncio.Semaphore,
    last_t:  list[float],
    entry:   Optional[str] = None,
    limit:   Optional[int] = None,
    verbose: bool = False,
) -> dict[str, str]:
    """
    Launch a one-level-DFS breadcrumb crawl from each entry point.

    The entry point itself is depth 0 — crawl_recursive reads its breadcrumb
    and uses it as the parent for its direct children. From there, a child is
    only followed if its breadcrumb is exactly one segment deeper and shares
    the parent's full prefix (see crawl_recursive docstring).

    Pages with no breadcrumb are logged to NO_BREADCRUMB_FILE, not scraped.

    Returns {url: category_label}.
    """
    discovered: dict[str, str] = {}
    no_crumb: set[str] = set()

    # Fresh no-breadcrumb log per run
    if NO_BREADCRUMB_FILE.exists():
        NO_BREADCRUMB_FILE.unlink()

    entries = ENTRY_POINTS
    if entry:
        entries = [(p, c) for p, c in ENTRY_POINTS if p == entry]
        if not entries:
            console.print(f"[red]Unknown entry point: {entry}[/red]")
            console.print(f"Valid options: {[p for p, _ in ENTRY_POINTS]}")
            return {}

    for path, base_category in entries:
        entry_url = BASE_URL + path
        console.print(
            f"\n[bold cyan]Entry point:[/bold cyan] {path} "
            f"(base category: {base_category})"
        )
        before = len(discovered)

        await crawl_recursive(
            client=client,
            sem=sem,
            last_t=last_t,
            url=entry_url,
            parent_crumb=[],          # entry point is the root; its own crumb is read inside
            base_category=base_category,
            discovered=discovered,
            no_crumb=no_crumb,
            limit=limit,
            verbose=verbose,
            depth=0,
        )

        added = len(discovered) - before
        console.print(f"  → {added} pages discovered under {path}")
        if limit is not None and len(discovered) >= limit:
            break

    console.print(
        f"\n[bold green]Discovery complete:[/bold green] "
        f"{len(discovered)} total URLs"
    )
    if no_crumb:
        console.print(
            f"[yellow]{len(no_crumb)} pages had no breadcrumb[/yellow] "
            f"→ logged to {NO_BREADCRUMB_FILE}"
        )
    console.print()
    by_cat: dict[str, int] = defaultdict(int)
    for cat in discovered.values():
        by_cat[cat] += 1
    for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        console.print(f"  {cat:12s}  {count:5d} URLs")

    return discovered


# ── Main scraper ──────────────────────────────────────────────────────────────

def _save_discovered(url_map: dict[str, str]) -> None:
    """Persist the discovery result so scraping can run later from cache."""
    DISCOVERED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DISCOVERED_FILE.open("w", encoding="utf-8") as f:
        for url, category in url_map.items():
            f.write(json.dumps({"url": url, "category": category}, ensure_ascii=False) + "\n")


def _load_discovered() -> dict[str, str]:
    """Load a previously saved discovery result."""
    url_map: dict[str, str] = {}
    if not DISCOVERED_FILE.exists():
        return url_map
    with DISCOVERED_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                url_map[obj["url"]] = obj["category"]
            except Exception:
                pass
    return url_map


async def scrape(
    limit:         Optional[int] = None,
    dry_run:       bool = False,
    entry:         Optional[str] = None,
    verbose:       bool = False,
    discover_only: bool = False,
    scrape_only:   bool = False,
) -> None:
    """
    Two-phase pipeline:

      Discovery phase (network): walk the breadcrumb DFS from each entry point,
        produce {url: category}, and save it to DISCOVERED_FILE.

      Scrape phase (cache): read DISCOVERED_FILE, parse each page from the HTML
        cache (no network needed if already cached), write to OUTPUT_FILE.

    Flags:
      --discover-only : run discovery, save the URL list, then stop.
      --scrape-only   : skip discovery, load the saved URL list, scrape from cache.
      (default)       : run discovery then scraping in one go.
      --dry-run       : run discovery, print a summary, scrape nothing.
    """
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    sem    = asyncio.Semaphore(1)
    last_t = [0.0]

    async with build_client() as client:
        # ── Phase 1: Discovery ────────────────────────────────────────────────
        if scrape_only:
            url_map = _load_discovered()
            if not url_map:
                console.print(
                    f"[red]No discovered URLs found at {DISCOVERED_FILE}.[/red] "
                    f"Run discovery first (without --scrape-only)."
                )
                return
            console.print(
                f"[bold cyan]Loaded {len(url_map)} discovered URLs from "
                f"{DISCOVERED_FILE}[/bold cyan]"
            )
        else:
            url_map = await discover_urls(
                client, sem, last_t, entry=entry, limit=limit, verbose=verbose
            )
            _save_discovered(url_map)
            console.print(f"[dim]Discovery saved → {DISCOVERED_FILE}[/dim]")

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

        if discover_only:
            console.print(
                f"\n[bold green]Discovery complete.[/bold green] "
                f"{len(url_map)} URLs saved to {DISCOVERED_FILE}. "
                f"Run with --scrape-only to scrape from cache."
            )
            return

        # ── Phase 2: Scrape ───────────────────────────────────────────────────
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
                        html = await fetch(client, url, sem, last_t, verbose=verbose)
                        page = parse_page(html, url, category)
                        if len(page.body_text) < 80:
                            if verbose:
                                console.print(f"  [yellow][skip — too short][/yellow] {url.split('/')[-1]}")
                            continue
                        out_f.write(
                            json.dumps(page.to_dict(), ensure_ascii=False) + "\n"
                        )
                        out_f.flush()
                        if verbose:
                            crumb_str = " / ".join(page.breadcrumb) if page.breadcrumb else "no breadcrumb"
                            console.print(
                                f"  [green][scraped][/green] {page.title[:50]}  "
                                f"[dim][{category}] {crumb_str}[/dim]"
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
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each page as it is fetched and scraped",
    )
    parser.add_argument(
        "--discover-only", action="store_true",
        help="Run discovery, save the URL list to discovered_urls.jsonl, then stop",
    )
    parser.add_argument(
        "--scrape-only", action="store_true",
        help="Skip discovery; load discovered_urls.jsonl and scrape from cache",
    )
    args = parser.parse_args()
    asyncio.run(scrape(
        limit=args.limit,
        dry_run=args.dry_run,
        entry=args.entry,
        verbose=args.verbose,
        discover_only=args.discover_only,
        scrape_only=args.scrape_only,
    ))


if __name__ == "__main__":
    main()