"""
data_pipeline/common.py

Shared building blocks for the Elden Ring wiki pipeline.

Both discover.py and scrape.py import from here:
  - configuration constants and file paths
  - the WikiPage dataclass
  - the HTTP client + on-disk cache (fetch)
  - URL helpers (to_absolute, is_wiki_page)
  - breadcrumb + title extraction
  - category inference from breadcrumb

Keeping these here means discovery and scraping never drift out of sync on
the things they both depend on (URL rules, cache layout, breadcrumb parsing).
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)
from rich.console import Console
from dotenv import load_dotenv

load_dotenv()
console = Console()

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL    = "https://eldenring.wiki.fextralife.com"
RATE_LIMIT  = float(os.getenv("WIKI_RATE_LIMIT_SECONDS", "0.5"))

DATA_DIR           = Path("data_pipeline/data")
CACHE_DIR          = DATA_DIR / "cache"
DISCOVERED_FILE    = DATA_DIR / "discovered_urls.jsonl"
NO_BREADCRUMB_FILE = DATA_DIR / "no_breadcrumb.txt"
OUTPUT_FILE        = DATA_DIR / "pages.jsonl"

USER_AGENT = (
    "EldenRingRAGBot/1.0 (portfolio project; educational use; "
    "respectful scraping at ~1 req/sec)"
)

# Safety net against unexpected cycles during recursive discovery.
MAX_DEPTH = 8

# ── Entry points ──────────────────────────────────────────────────────────────
# The four top-level nav sections. Each is the root of one discovery tree.
# The trailing test entry (/Caelid) is handy for validating a single region.

ENTRY_POINTS: list[tuple[str, str]] = [
    ("/World+Information",     "location"),  # Locations, NPCs, Bosses, Lore, Merchants
    ("/Equipment+&+Magic",     "item"),       # Weapons, Armor, Spells, Talismans, etc.
    ("/Character+Information", "lore"),        # Classes, Stats, Status Effects
    ("/Guides+&+Walkthroughs", "quest"),       # Walkthrough, Side Quests, Endings, Crafting
    # Test / partial entry point (use via --entry):
    ("/Caelid",                "location"),    # single region, for testing the DFS
]

# ── Category inference ────────────────────────────────────────────────────────
# Maps breadcrumb segment keywords to a RAG category. Ordered most-specific
# to least-specific; the first match (scanning the breadcrumb from the deepest
# segment up) wins.

BREADCRUMB_CATEGORY_MAP: list[tuple[str, str]] = [
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

# ── URL filtering ─────────────────────────────────────────────────────────────

SKIP_PATHS: frozenset[str] = frozenset({
    "/Interactive+Map", "/Maps", "/Elden+Ring+Wiki", "/Patch+Notes",
    "/Controls", "/Multiplayer+Coop+and+Online", "/Online+Information",
    "/PvP", "/PvE+Builds", "/PvP+Builds", "/Builds", "/Player+Trade",
    "/Build+Calculator", "/Summon+Range+Calculator", "/New+Game+Plus",
    "/Nightreign", "/Elden+Ring+Movie", "/Elden+Ring+Tarnished+Edition",
    "/Multiplayer+Items", "/Gestures", "/Rebirth", "/todo",
    "/Trophy+&+Achievement+Guide", "/Wiki+Shop", "/Covenants",
})

SKIP_URL_FRAGMENTS: tuple[str, ...] = (
    "action=", "/Shop", "/forums", "/news", "/blog", "?", "#",
)

# Other fextralife wikis that may appear as cross-links (checked on netloc).
SKIP_DOMAINS: frozenset[str] = frozenset({
    "fextralife.com",
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
    breadcrumb:        list[str]          # ["World Information", "Locations", "Caelid"]
    body_text:         str
    infobox:           dict[str, str]
    dialogue:          list[str]
    item_descriptions: list[str]
    image_url:         Optional[str]
    internal_links:    list[str]
    scraped_at:        float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# ── HTTP client + retry ───────────────────────────────────────────────────────

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


# ── On-disk cache ─────────────────────────────────────────────────────────────

def cache_path(url: str) -> Path:
    return CACHE_DIR / f"{hashlib.md5(url.encode()).hexdigest()}.html"


def read_cache(url: str) -> Optional[str]:
    p = cache_path(url)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def write_cache(url: str, html: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(url).write_text(html, encoding="utf-8")


async def fetch(
    client:  httpx.AsyncClient,
    url:     str,
    sem:     asyncio.Semaphore,
    last_t:  list[float],
    verbose: bool = False,
) -> str:
    """Fetch a URL, served from cache when present, rate-limited otherwise."""
    cached = read_cache(url)
    if cached is not None:
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
    write_cache(url, html)
    return html


def is_cached(url: str) -> bool:
    """True if the URL's HTML is already on disk (no network needed)."""
    return cache_path(url).exists()


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
    return href.split("#")[0].rstrip("/") or None


def is_wiki_page(url: str) -> bool:
    """True if this is a scrape-worthy Elden Ring wiki content URL."""
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.netloc != "eldenring.wiki.fextralife.com":
        return False
    if parsed.netloc in SKIP_DOMAINS:
        return False
    path = parsed.path.rstrip("/")
    if not path or path == "/Elden+Ring+Wiki":
        return False
    if path in SKIP_PATHS:
        return False
    # Reject image / file URLs
    if path.startswith("/file/") or path.lower().endswith(
        (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".pdf")
    ):
        return False
    if parsed.query:
        return False
    if any(frag in url for frag in SKIP_URL_FRAGMENTS):
        return False
    return True


# ── Breadcrumb + title extraction ─────────────────────────────────────────────

def extract_breadcrumb(soup: BeautifulSoup) -> list[str]:
    """
    Extract the breadcrumb trail from:
        <div id="breadcrumbs-container">
          <a href="...">World Information</a>
          <a href="...">Locations</a>
          <div id="breadcrumbs-bcontainer" style="display:none;">
            <a href="#" id="btnCreateBreadcrumb">+</a>   <!-- editor button -->
          </div>
        </div>

    Returns e.g. ["World Information", "Locations", "Caelid"].
    Returns [] when the page has no breadcrumb (top-level hub pages).
    """
    container = soup.find(id="breadcrumbs-container")
    if not container:
        return []

    # Drop the hidden editor sub-div (the '+' button)
    editor_div = container.find(id="breadcrumbs-bcontainer")
    if editor_div:
        editor_div.decompose()

    segments: list[str] = []
    for a in container.find_all("a"):
        href = a.get("href", "")
        text = a.get_text(strip=True)
        if not text or len(text) <= 1:
            continue
        if text in ("Elden Ring Wiki", ">", "/", "»", "+"):
            continue
        if href in ("#", ""):
            continue
        segments.append(text)

    return segments


def extract_title(soup: BeautifulSoup) -> str:
    """Page title, with the '| Elden Ring Wiki' suffix stripped."""
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


def infer_category(breadcrumb: list[str], default: str) -> str:
    """
    Assign a RAG category from the breadcrumb.
    Scans segments from the deepest up; first keyword match wins.
    Falls back to `default` if nothing matches.
    """
    for segment in reversed(breadcrumb):
        for keyword, category in BREADCRUMB_CATEGORY_MAP:
            if keyword.lower() in segment.lower():
                return category
    return default