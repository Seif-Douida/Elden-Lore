"""
data_pipeline/debug_scraper.py

Standalone debug script — run this INSTEAD of the full scraper to diagnose
why crawl_recursive stops after the entry point page.

Run with:
    uv run python data_pipeline/debug_scraper.py

It will:
  1. Fetch /World+Information
  2. Print the raw breadcrumb HTML found
  3. Print what extract_breadcrumb() returns
  4. Print the first 20 content links found by extract_content_links()
  5. Fetch the first child link and print its breadcrumb
  6. Run breadcrumb_extends() and show the result
  7. Fetch /Locations and repeat the same checks
"""

import asyncio
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import httpx

BASE_URL  = "https://eldenring.wiki.fextralife.com"
CACHE_DIR = Path("data_pipeline/data/cache")

USER_AGENT = (
    "EldenRingRAGBot/1.0 (portfolio project; educational use; "
    "respectful scraping at ~1 req/sec)"
)

SKIP_URL_FRAGMENTS = ("action=", "/Shop", "/forums", "/news", "/blog", "?", "#")

SKIP_PATHS = frozenset({
    "/Interactive+Map", "/Maps", "/Elden+Ring+Wiki", "/Patch+Notes",
    "/Controls", "/Multiplayer+Coop+and+Online", "/PvP", "/Builds",
})


# ── Minimal cache ─────────────────────────────────────────────────────────────

def _cache_path(url: str) -> Path:
    return CACHE_DIR / f"{hashlib.md5(url.encode()).hexdigest()}.html"

def _read_cache(url: str):
    p = _cache_path(url)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None

def _write_cache(url: str, html: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(url).write_text(html, encoding="utf-8")

async def fetch(client, url):
    cached = _read_cache(url)
    if cached:
        print(f"  [cache hit] {url}")
        return cached
    print(f"  [fetching] {url}")
    resp = await client.get(url)
    resp.raise_for_status()
    html = resp.text
    _write_cache(url, html)
    return html


# ── Copies of the two suspect functions ──────────────────────────────────────

def extract_breadcrumb(soup: BeautifulSoup) -> list[str]:
    container = soup.find(id="breadcrumbs-container")
    if not container:
        return []
    # Remove the hidden editor button div before parsing
    editor = container.find(id="breadcrumbs-bcontainer")
    if editor:
        editor.decompose()
    segments = []
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


def to_absolute(href: str):
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
    if not url:
        return False
    parsed = urlparse(url)
    # Must be exactly the Elden Ring wiki subdomain
    if parsed.netloc != "eldenring.wiki.fextralife.com":
        return False
    path = parsed.path.rstrip("/")
    if not path or path == "/Elden+Ring+Wiki":
        return False
    if path in SKIP_PATHS:
        return False
    if parsed.query:
        return False
    if any(frag in url for frag in SKIP_URL_FRAGMENTS):
        return False
    return True


def extract_content_links(soup: BeautifulSoup) -> list[str]:
    seen: set[str] = set()
    links: list[str] = []
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


def breadcrumb_extends(parent: list[str], child: list[str]) -> bool:
    if not parent:
        return True
    if len(child) != len(parent) + 1:
        return False
    return child[:len(parent)] == parent


# ── Debug routine ─────────────────────────────────────────────────────────────

async def debug():
    client = httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT, "Accept": "text/html"},
        follow_redirects=True,
        timeout=20.0,
    )

    async with client:
        for test_url in [
            f"{BASE_URL}/World+Information",
            f"{BASE_URL}/Locations",
        ]:
            sep = "=" * 70
            print(f"\n{sep}")
            print(f"TESTING: {test_url}")
            print(sep)

            html = await fetch(client, test_url)
            soup = BeautifulSoup(html, "lxml")

            # ── 1. Raw breadcrumb HTML ────────────────────────────────────
            print("\n[1] RAW BREADCRUMB HTML:")
            container = soup.find(id="breadcrumbs-container")
            if container:
                print(str(container)[:800])
            else:
                print("  !! No element found with id='breadcrumbs-container'")
                # Search for any element containing breadcrumb-like text
                print("\n  Searching for alternative breadcrumb elements...")
                for candidate_id in ("breadcrumb", "breadcrumbs", "crumbs",
                                     "page-breadcrumb", "wiki-breadcrumb"):
                    el = soup.find(id=candidate_id)
                    if el:
                        print(f"  Found id='{candidate_id}': {str(el)[:300]}")
                for candidate_class in ("breadcrumb", "breadcrumbs",
                                        "wiki-breadcrumb", "crumbs"):
                    els = soup.find_all(class_=candidate_class)
                    if els:
                        print(f"  Found class='{candidate_class}' "
                              f"({len(els)} elements): {str(els[0])[:300]}")
                # Also dump all IDs present in the page for inspection
                all_ids = [el.get("id") for el in soup.find_all(id=True)]
                print(f"\n  All IDs in page ({len(all_ids)} total):")
                for pid in all_ids[:40]:
                    print(f"    #{pid}")
                if len(all_ids) > 40:
                    print(f"    ... and {len(all_ids)-40} more")

            # ── 2. extract_breadcrumb() result ────────────────────────────
            print("\n[2] extract_breadcrumb() result:")
            crumb = extract_breadcrumb(soup)
            print(f"  {crumb!r}")

            # ── 3. Content block presence ─────────────────────────────────
            print("\n[3] Content block selector check:")
            for sel in ("div#wiki-content-block", "div.wiki-content",
                        "article", "main"):
                el = soup.select_one(sel)
                print(f"  {sel!r:40s} → {'FOUND' if el else 'not found'}")

            # ── 4. Content links ──────────────────────────────────────────
            print("\n[4] extract_content_links() — first 20 results:")
            links = extract_content_links(soup)
            print(f"  Total links found: {len(links)}")
            for lnk in links[:20]:
                print(f"    {lnk}")
            if not links:
                # Diagnose: how many raw <a> tags in the content block?
                content = soup.select_one(
                    "div#wiki-content-block, div.wiki-content, article"
                )
                if content:
                    all_a = content.find_all("a", href=True)
                    print(f"  Raw <a> tags in content block: {len(all_a)}")
                    print("  First 10 raw hrefs:")
                    for a in all_a[:10]:
                        print(f"    href={a['href']!r}  "
                              f"→ abs={to_absolute(a['href'])!r}  "
                              f"is_wiki={is_wiki_page(to_absolute(a['href']) or '')}")
                else:
                    all_a = soup.find_all("a", href=True)
                    print(f"  No content block found. Raw <a> tags in full soup: {len(all_a)}")
                    print("  First 10 raw hrefs:")
                    for a in all_a[:10]:
                        print(f"    href={a['href']!r}")

            # ── 5. If links found, fetch first child and check breadcrumb ─
            if links:
                child_url = links[0]
                print(f"\n[5] Fetching first child link: {child_url}")
                try:
                    child_html = await fetch(client, child_url)
                    child_soup = BeautifulSoup(child_html, "lxml")
                    child_crumb = extract_breadcrumb(child_soup)
                    print(f"  Parent breadcrumb : {crumb!r}")
                    print(f"  Child breadcrumb  : {child_crumb!r}")
                    result = breadcrumb_extends(crumb, child_crumb)
                    print(f"  breadcrumb_extends: {result}")
                    if not result:
                        print(f"  WHY: parent len={len(crumb)}, "
                              f"child len={len(child_crumb)}, "
                              f"need child len={len(crumb)+1}")
                        if child_crumb:
                            print(f"  child[:len(parent)] = {child_crumb[:len(crumb)]!r}")
                            print(f"  parent              = {crumb!r}")
                            print(f"  match: {child_crumb[:len(crumb)] == crumb}")
                except Exception as e:
                    print(f"  Error fetching child: {e}")


if __name__ == "__main__":
    asyncio.run(debug())