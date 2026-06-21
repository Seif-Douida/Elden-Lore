"""
data_pipeline/wiki_scraper.py

Async Elden Ring wiki scraper (fextralife).

Scope: Everything useful for a RAG chatbot —
  lore, bosses, NPCs + dialogue, quests, items (weapons/armor/talismans/
  ashes/sorceries/incantations/spirits/shields/key items/cookbooks),
  locations, sites of grace, enemies, merchants, DLC content.

Strategy:
  1. Seed crawl — fetch 30+ category index pages, collect entity URLs.
  2. For categories that don't have a single index page (e.g. individual
     NPC quest pages), follow internal links one level deeper.
  3. For each entity page: extract title, body text, infobox, image URL,
     NPC dialogue blocks, and item description blocks.
  4. Cache everything to disk — safe to interrupt and resume.

Output: data_pipeline/data/pages.jsonl  (one JSON object per line)

Usage:
    uv run python data_pipeline/wiki_scraper.py --dry-run
    uv run python data_pipeline/wiki_scraper.py --limit 20
    uv run python data_pipeline/wiki_scraper.py            # full run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
)
from dotenv import load_dotenv
import os

load_dotenv()
console = Console()

# ── Configuration ────────────────────────────────────────────────────────────

BASE_URL   = "https://eldenring.wiki.fextralife.com"
RATE_LIMIT = float(os.getenv("WIKI_RATE_LIMIT_SECONDS", "1.2"))
CACHE_DIR  = Path("data_pipeline/data/cache")
OUTPUT_FILE = Path("data_pipeline/data/pages.jsonl")
USER_AGENT = (
    "EldenRingRAGBot/1.0 (portfolio project; educational use; "
    "respectful scraping at ~1 req/sec)"
)

# ── Category seeds ───────────────────────────────────────────────────────────
# (path, category_label, follow_links_deeper)
#
# follow_links_deeper=True means we also crawl entity pages discovered
# from this index and follow *their* internal wiki links one more level.
# Use this for categories where the good content is on sub-pages
# (e.g. individual NPC pages, individual item pages).

CATEGORY_SEEDS: list[tuple[str, str, bool]] = [
    # ── Lore ──────────────────────────────────────────────────────────────
    ("/Lore",                           "lore",     False),
    ("/Timeline+of+Lands+Between+history", "lore",  False),
    ("/Characters",                     "lore",     False),

    # ── Bosses & Enemies ──────────────────────────────────────────────────
    ("/Bosses",                         "boss",     False),
    ("/Creatures+and+Enemies",          "boss",     False),

    # ── Items ─────────────────────────────────────────────────────────────
    ("/Weapons",                        "item",     False),
    ("/Armor",                          "item",     False),
    ("/Talismans",                      "item",     False),
    ("/Ashes+of+War",                   "item",     False),
    ("/Sorceries",                      "item",     False),
    ("/Incantations",                   "item",     False),
    ("/Spirit+Ashes",                   "item",     False),
    ("/Shields",                        "item",     False),
    ("/Key+Items",                      "item",     False),
    ("/Consumables",                    "item",     False),
    ("/Cookbooks",                      "item",     False),
    ("/Upgrade+Materials",              "item",     False),
    ("/Tools",                          "item",     False),
    ("/Arrows+and+Bolts",               "item",     False),

    # ── NPCs, Quests & Dialogue ───────────────────────────────────────────
    # follow_links_deeper=True: individual NPC pages have dialogue + quest steps
    ("/NPCs",                           "quest",    True),
    ("/Side+Quests",                    "quest",    True),
    ("/Merchants",                      "quest",    False),

    # ── Locations & Sites of Grace ────────────────────────────────────────
    ("/Locations",                      "location", False),
    ("/Sites+of+Grace",                 "location", False),
    ("/Legacy+Dungeons",                "location", False),
    ("/Maps",                           "location", False),

    # ── DLC – Shadow of the Erdtree ───────────────────────────────────────
    ("/Shadow+of+the+Erdtree",          "lore",     False),
    ("/DLC+Items",                      "item",     False),
    ("/DLC+Weapons",                    "item",     False),
    ("/DLC+Armor",                      "item",     False),
    ("/DLC+Bosses",                     "boss",     False),
    ("/DLC+NPCs",                       "quest",    True),
]

# Paths to skip when harvesting links from index pages
SKIP_PATH_PATTERNS: tuple[str, ...] = (
    "/Interactive+Map",
    "/Elden+Ring+Wiki",
    "/General+Information",
    "/Patch+Notes",
    "/Controls",
    "/Multiplayer",
    "/New+Game+Plus",
    "/Game+Progress+Route",
    "/Guides+and+Walkthroughs",
    "/Builds",
    "/Build+Calculator",
    "/Player+Trade",
    "/PvP",
    "?",
    "#",
)

# CSS selectors for noise elements to strip before extracting text
NOISE_SELECTORS: list[str] = [
    "div.wiki-commentslist",
    "div#wiki-commentslist",
    "div.comments",
    "div#comments",
    "div#wiki-content-block > div.col-sm-3",   # right sidebar
    "nav",
    "div.wiki-nav",
    "div#header",
    "div#footer",
    "footer",
    "header",
    "div.ads",
    "div.adsbygoogle",
    "div[class*='ad-']",
    "div[class*='social']",
    "span.wiki-edit",
    "div.wiki-page-edit",
    "a[href*='action=edit']",
    "div.wiki-page-discussion",
    "div.wiki-rating",
    "div[class*='rating']",
    "div#wiki-content-block div.toc",
]

# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class WikiPage:
    url:            str
    title:          str
    category:       str
    body_text:      str                 # clean prose text
    infobox:        dict[str, str]      # structured stats / metadata
    dialogue:       list[str]           # NPC dialogue lines (if any)
    item_descriptions: list[str]        # in-game item description text (if any)
    image_url:      Optional[str]       # primary image CDN URL
    internal_links: list[str]           # other wiki URLs linked from this page
    scraped_at:     float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


# ── HTTP client ───────────────────────────────────────────────────────────────

def build_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={
            "User-Agent": USER_AGENT,
            "Accept":     "text/html,application/xhtml+xml",
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
async def fetch_html(client: httpx.AsyncClient, url: str) -> str:
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.text


# ── Cache helpers ─────────────────────────────────────────────────────────────

def cache_path(url: str) -> Path:
    key = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{key}.html"


def read_cache(url: str) -> Optional[str]:
    p = cache_path(url)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def write_cache(url: str, html: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(url).write_text(html, encoding="utf-8")


async def fetch_with_cache(
    client: httpx.AsyncClient,
    url: str,
    rate_semaphore: asyncio.Semaphore,
    last_request_time: list[float],
) -> str:
    cached = read_cache(url)
    if cached:
        return cached

    async with rate_semaphore:
        elapsed = time.monotonic() - last_request_time[0]
        if elapsed < RATE_LIMIT:
            await asyncio.sleep(RATE_LIMIT - elapsed)
        html = await fetch_html(client, url)
        last_request_time[0] = time.monotonic()

    write_cache(url, html)
    return html


# ── HTML parsing ──────────────────────────────────────────────────────────────

def strip_noise(soup: BeautifulSoup) -> None:
    """Remove all noisy elements from the soup in-place."""
    for sel in NOISE_SELECTORS:
        for el in soup.select(sel):
            el.decompose()
    for tag in ("script", "style", "iframe", "noscript"):
        for el in soup.find_all(tag):
            el.decompose()


def clean_title(raw: str) -> str:
    """Strip the '| Elden Ring Wiki' suffix that appears on h1 and <title> tags."""
    return raw.split("|")[0].strip()


def extract_title(soup: BeautifulSoup) -> str:
    # FIX: apply clean_title to h1 as well — fextralife h1 includes the suffix
    h1 = soup.find("h1")
    if h1:
        return clean_title(h1.get_text(strip=True))
    title_tag = soup.find("title")
    if title_tag:
        return clean_title(title_tag.get_text(strip=True))
    return "Unknown"


def extract_image_url(soup: BeautifulSoup) -> Optional[str]:
    """Find the primary entity image. Returns an absolute URL or None."""
    for selector in [
        "div.infobox img",
        "div#wiki-content-block img",
        "table.wiki_table img",
    ]:
        for img in soup.select(selector):
            src = img.get("src", "")
            if not src:
                continue
            src_lower = src.lower()
            if "icon" in src_lower or "logo" in src_lower or "banner" in src_lower:
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


# FIX: infobox — reject any cell that is itself a diamond-separated nav list
def _cell_is_nav_list(text: str) -> bool:
    return "♦" in text and text.count("♦") >= 2


def extract_infobox(soup: BeautifulSoup) -> dict[str, str]:
    """
    Parse the stats/infobox table.
    Skips cells that are themselves diamond nav lists (sidebar location tables).
    """
    infobox: dict[str, str] = {}
    tables = soup.select("table.wiki_table, table.infobox")
    for table in tables[:2]:
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) == 2:
                key = cells[0].get_text(strip=True)
                val = cells[1].get_text(strip=True)
                # FIX: skip if key or value is a nav link list
                if key and val and len(key) < 80:
                    if not _cell_is_nav_list(key) and not _cell_is_nav_list(val):
                        infobox[key] = val
            elif len(cells) == 1:
                header = cells[0].get_text(strip=True)
                if header and not _cell_is_nav_list(header):
                    infobox["_section"] = header
    return infobox


# ── Dialogue extraction ───────────────────────────────────────────────────────
#
# Fextralife renders NPC dialogue inside <blockquote> tags or in table cells
# that contain italicised text. We collect all of these.
#
_DIALOGUE_MIN_LEN = 20   # ignore very short fragments

def extract_dialogue(soup: BeautifulSoup) -> list[str]:
    """Extract NPC dialogue lines from blockquotes and italic table cells."""
    lines: list[str] = []
    seen: set[str] = set()

    # Blockquotes are the most reliable container
    for bq in soup.find_all("blockquote"):
        text = bq.get_text(separator=" ", strip=True)
        if len(text) >= _DIALOGUE_MIN_LEN and text not in seen:
            seen.add(text)
            lines.append(text)

    # Some dialogue is in <em>/<i> inside table cells
    for cell in soup.select("td em, td i"):
        text = cell.get_text(separator=" ", strip=True)
        if len(text) >= _DIALOGUE_MIN_LEN and text not in seen:
            seen.add(text)
            lines.append(text)

    return lines


# ── Item description extraction ───────────────────────────────────────────────
#
# In-game item descriptions are rendered inside <div class="codex"> or inside
# a <p> immediately following a heading that contains "Description" or "Lore".
# They are short, flavour-text paragraphs and are extremely valuable for RAG.

def extract_item_descriptions(soup: BeautifulSoup) -> list[str]:
    """Extract in-game item description / lore flavour text blocks."""
    descriptions: list[str] = []
    seen: set[str] = set()

    # Method 1: dedicated .codex divs (Fextralife wraps item text in these)
    for div in soup.select("div.codex, div.item-desc, p.description"):
        text = div.get_text(separator=" ", strip=True)
        if len(text) >= 30 and text not in seen:
            seen.add(text)
            descriptions.append(text)

    # Method 2: <p> that immediately follows a heading with "Description" or "Lore"
    for heading in soup.find_all(["h2", "h3", "h4"]):
        heading_text = heading.get_text(strip=True).lower()
        if "description" in heading_text or "lore" in heading_text or "flavour" in heading_text:
            sibling = heading.find_next_sibling()
            while sibling and sibling.name in ("p", "div"):
                text = sibling.get_text(separator=" ", strip=True)
                if len(text) >= 30 and text not in seen:
                    seen.add(text)
                    descriptions.append(text)
                sibling = sibling.find_next_sibling()

    return descriptions


# ── Body text extraction ──────────────────────────────────────────────────────

# FIX: smarter diamond detection — only strip blocks that look like nav lists
# (sequences of proper-noun-like phrases separated by ♦)
# The Timeline page uses ♦ as a decorative separator in lore text — we keep those.
_NAV_DIAMOND_RE = re.compile(
    r"^([A-Z][^\n♦]{2,60}♦\s*){2,}",   # 2+ title-case phrases separated by ♦
)

def _is_nav_diamond_block(text: str) -> bool:
    """
    Returns True only for navigation link lists like:
        'Abandoned Cave ♦ Academy Crystal Cave ♦ Aeonia Swamp ♦ ...'
    Returns False for decorative lore use like:
        '§ ~ ~ ~ ♦ ~ ~ ~ § ... Time immemorial'
    """
    if "♦" not in text or text.count("♦") < 2:
        return False
    # If most non-♦ segments start with a capital letter, it's a nav list
    segments = [s.strip() for s in text.split("♦") if s.strip()]
    if len(segments) < 2:
        return False
    capitalised = sum(1 for s in segments if s and s[0].isupper())
    return (capitalised / len(segments)) > 0.7


def extract_body_text(soup: BeautifulSoup) -> str:
    """Extract clean article prose, filtering nav lists and trivial lines."""
    content = soup.select_one(
        "div#wiki-content-block, div.wiki-content, article, div[role='main']"
    )
    if not content:
        content = soup.find("body")
    if not content:
        return ""

    lines: list[str] = []
    for el in content.find_all(["p", "li", "h2", "h3", "h4", "td"]):
        text = el.get_text(separator=" ", strip=True)
        if len(text) < 20:
            continue
        # FIX: only strip nav-style diamond lists, not decorative lore use
        if _is_nav_diamond_block(text):
            continue
        lines.append(text)

    raw = "\n".join(lines)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def extract_internal_links(soup: BeautifulSoup) -> list[str]:
    seen: set[str] = set()
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        if any(p in href for p in SKIP_PATH_PATTERNS):
            continue
        if href.startswith("http") and BASE_URL not in href:
            continue
        if href.startswith(("mailto:", "javascript:")):
            continue
        if href.startswith("/"):
            href = BASE_URL + href
        elif not href.startswith("http"):
            href = urljoin(BASE_URL, href)
        parsed = urlparse(href)
        if parsed.netloc and "fextralife" not in parsed.netloc:
            continue
        href = href.split("#")[0].rstrip("/")
        if href not in seen and len(urlparse(href).path.strip("/")) > 2:
            seen.add(href)
            links.append(href)
    return links


def parse_page(html: str, url: str, category: str) -> WikiPage:
    soup = BeautifulSoup(html, "lxml")
    strip_noise(soup)
    return WikiPage(
        url=url,
        title=extract_title(soup),
        category=category,
        body_text=extract_body_text(soup),
        infobox=extract_infobox(soup),
        dialogue=extract_dialogue(soup),
        item_descriptions=extract_item_descriptions(soup),
        image_url=extract_image_url(soup),
        internal_links=extract_internal_links(soup),
    )


# ── URL discovery ─────────────────────────────────────────────────────────────

async def _links_from_page(html: str, category: str) -> dict[str, str]:
    """Return all wiki entity links found on a page as {url: category}."""
    soup = BeautifulSoup(html, "lxml")
    content = soup.select_one("div#wiki-content-block, div.wiki-content") or soup
    found: dict[str, str] = {}
    for a in content.find_all("a", href=True):
        href: str = a["href"]
        if not href.startswith("/"):
            continue
        if any(p in href for p in SKIP_PATH_PATTERNS):
            continue
        slug = href.strip("/").split("#")[0]
        if len(slug) < 3:
            continue
        full_url = BASE_URL + "/" + slug
        if full_url not in found:
            found[full_url] = category
    return found


async def discover_urls(
    client: httpx.AsyncClient,
    rate_semaphore: asyncio.Semaphore,
    last_request_time: list[float],
) -> dict[str, str]:
    """
    Crawl every seed page.
    For seeds with follow_links_deeper=True, also crawl each discovered
    entity page and collect its internal links (one extra hop).
    Returns {url: category_label}.
    """
    discovered: dict[str, str] = {}
    console.print("[bold cyan]Discovering URLs from category seeds...[/bold cyan]")

    for path, category, follow_deeper in CATEGORY_SEEDS:
        index_url = BASE_URL + path
        try:
            html = await fetch_with_cache(
                client, index_url, rate_semaphore, last_request_time
            )
        except httpx.HTTPStatusError as e:
            console.print(f"  [red]✗[/red] {path}: HTTP {e.response.status_code}")
            continue
        except Exception as e:
            console.print(f"  [red]✗[/red] {path}: {e}")
            continue

        first_level = await _links_from_page(html, category)
        count_before = len(discovered)
        for url, cat in first_level.items():
            if url not in discovered:
                discovered[url] = cat

        # For NPC/quest seeds: follow each discovered page one level deeper
        # to pick up individual quest walkthroughs, dialogue pages, etc.
        if follow_deeper:
            deeper: dict[str, str] = {}
            for entity_url in list(first_level.keys())[:200]:  # cap at 200 to avoid explosion
                try:
                    e_html = await fetch_with_cache(
                        client, entity_url, rate_semaphore, last_request_time
                    )
                    for url2, cat2 in (await _links_from_page(e_html, category)).items():
                        if url2 not in discovered and url2 not in deeper:
                            deeper[url2] = cat2
                except Exception:
                    pass
            for url, cat in deeper.items():
                if url not in discovered:
                    discovered[url] = cat

        added = len(discovered) - count_before
        suffix = " [deeper crawl]" if follow_deeper else ""
        console.print(f"  [green]✓[/green] {path} ({category}){suffix} → +{added} URLs")

    console.print(f"\n[bold]Total URLs to scrape:[/bold] {len(discovered)}")
    return discovered


# ── Main scraper ──────────────────────────────────────────────────────────────

async def scrape(limit: Optional[int] = None, dry_run: bool = False) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rate_semaphore = asyncio.Semaphore(1)
    last_request_time: list[float] = [0.0]

    async with build_client() as client:
        url_map = await discover_urls(client, rate_semaphore, last_request_time)
        urls = list(url_map.items())
        if limit:
            urls = urls[:limit]

        if dry_run:
            console.print(f"\n[yellow]Dry run — would scrape {len(urls)} pages.[/yellow]")
            # Show a sample per category
            from collections import defaultdict
            by_cat: dict = defaultdict(list)
            for url, cat in urls:
                by_cat[cat].append(url)
            for cat, cat_urls in by_cat.items():
                console.print(f"\n  [bold]{cat}[/bold] ({len(cat_urls)} pages):")
                for u in cat_urls[:5]:
                    console.print(f"    {u}")
                if len(cat_urls) > 5:
                    console.print(f"    ... and {len(cat_urls)-5} more")
            return

        console.print(f"\n[bold cyan]Scraping {len(urls)} pages...[/bold cyan]")

        # Resume support
        already_scraped: set[str] = set()
        if OUTPUT_FILE.exists():
            with OUTPUT_FILE.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        already_scraped.add(json.loads(line)["url"])
                    except Exception:
                        pass
            if already_scraped:
                console.print(f"  Resuming: {len(already_scraped)} pages already done.")

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
                        task,
                        advance=1,
                        description=f"[cyan]{url.split('/')[-1][:45]}",
                    )
                    if url in already_scraped:
                        continue
                    try:
                        html = await fetch_with_cache(
                            client, url, rate_semaphore, last_request_time,
                        )
                        page = parse_page(html, url, category)
                        if len(page.body_text) < 80:
                            continue  # skip stub pages
                        out_f.write(
                            json.dumps(page.to_dict(), ensure_ascii=False) + "\n"
                        )
                        out_f.flush()
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code != 404:
                            errors.append((url, f"HTTP {e.response.status_code}"))
                    except Exception as e:
                        errors.append((url, str(e)))

    # Final count
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
    parser = argparse.ArgumentParser(description="Elden Ring wiki scraper")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover URLs only, print summary, do not scrape")
    parser.add_argument("--limit", type=int, default=None,
                        help="Scrape at most N pages (for testing)")
    args = parser.parse_args()
    asyncio.run(scrape(limit=args.limit, dry_run=args.dry_run))


if __name__ == "__main__":
    main()