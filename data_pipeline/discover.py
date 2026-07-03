"""
data_pipeline/discover.py

Phase 1 of the pipeline: DISCOVERY.

Walks the Elden Ring wiki using a one-level breadcrumb DFS from each entry
point and produces a clean manifest of URLs to scrape.

The rule: from a page whose breadcrumb is P, follow a linked page only if its
own breadcrumb C satisfies
      len(C) == len(P) + 1            (exactly one level deeper)
  AND C[:len(P)] == P                 (same path prefix)
Then C becomes the parent for that page's own children.

This keeps each page in exactly one tree (no cross-category leakage, no
duplicates) and naturally includes all DLC content, since DLC pages live
under the same nav sections.

Pages with no breadcrumb are logged to no_breadcrumb.txt and skipped.

Output: data/discovered_urls.jsonl  — one {"url", "category"} object per line.
This file is the input to scrape.py.

Usage:
    uv run python data_pipeline/discover.py
    uv run python data_pipeline/discover.py --entry /Caelid --verbose
    uv run python data_pipeline/discover.py --limit 100 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from common import (
    BASE_URL, MAX_DEPTH, ENTRY_POINTS,
    DISCOVERED_FILE, NO_BREADCRUMB_FILE,
    console, build_client, fetch,
    to_absolute, is_wiki_page,
    extract_breadcrumb, infer_category,
)


# ── Link extraction (discovery-time) ──────────────────────────────────────────

# Footer navigation tables (the big alphabetical "all weapons / all locations /
# all NPCs" blocks repeated at the bottom of every page) separate their links with a
# diamond bullet (♦). Meaningful CONTENT tables — the per-category item lists that are
# the only link path to some pages (e.g. the Katanas table linking Moonveil) — do not.
# We use that marker to strip only the footer navboxes.
_NAV_TABLE_MARKERS = ("♦", "◆")  # U+2666, U+25C6


def _is_footer_nav_table(table) -> bool:
    """A footer index/nav table, identified by the diamond-bullet link separator."""
    text = table.get_text()
    return any(m in text for m in _NAV_TABLE_MARKERS)


def extract_links_for_discovery(soup: BeautifulSoup) -> list[str]:
    """
    Collect wiki content links from a page, EXCLUDING the footer navigation tables.

    The footer nav tables (alphabetical "all weapons / all locations / all NPCs"
    blocks repeated at the bottom of every page) link to every sibling and cause
    thousands of fetch-then-reject cycles during DFS, so we strip them. They are
    identified by a diamond-bullet (♦) separator between links.

    We strip ONLY those (plus per-page infoboxes). Earlier this removed ALL
    wiki_table/sortable tables, which also dropped meaningful CONTENT tables (e.g.
    the Katanas list), so item pages reachable only through them — Moonveil,
    Dragonscale Blade, Serpentbone Blade… — never entered the corpus.
    """
    # Per-page infoboxes are stat sidebars, never a discovery path — always drop.
    for table in soup.select("table.infobox"):
        table.decompose()
    # Among wiki_table/sortable, drop only the ♦-marked footer navboxes; keep
    # meaningful content tables so their item links are harvested.
    for table in soup.select("table.wiki_table, table.sortable"):
        if _is_footer_nav_table(table):
            table.decompose()

    seen: set[str] = set()
    links: list[str] = []

    content = soup.select_one("div#wiki-content-block, div.wiki-content, article")
    if content:
        candidates = content.find_all("a", href=True)
    else:
        # Fallback: full soup, skipping nav/header/footer links
        nav_ids = set()
        for nav in soup.select("nav, header, div#header, div.wiki-nav, footer"):
            nav_ids.update(id(el) for el in nav.find_all("a"))
        candidates = [a for a in soup.find_all("a", href=True) if id(a) not in nav_ids]

    for a in candidates:
        url = to_absolute(a["href"])
        if url and is_wiki_page(url) and url not in seen:
            seen.add(url)
            links.append(url)
    return links


# ── No-breadcrumb logging ─────────────────────────────────────────────────────

def _note_no_breadcrumb(url: str, no_crumb: set[str]) -> None:
    """Track a URL that has no breadcrumb (in memory; written once at the end)."""
    no_crumb.add(url)


# ── Recursive crawler ─────────────────────────────────────────────────────────

async def crawl(
    client:        httpx.AsyncClient,
    sem:           asyncio.Semaphore,
    last_t:        list[float],
    url:           str,
    parent_crumb:  list[str],
    base_category: str,
    discovered:    dict[str, str],
    visited:       set[str],         # per-RUN cycle guard (separate from the output)
    no_crumb:      set[str],
    progress:      dict,             # {"next_milestone": int} — periodic log state
    limit_target:  Optional[int] = None,  # absolute size of `discovered` to stop at
    verbose:       bool = False,
    depth:         int = 0,
) -> None:
    """One-level breadcrumb DFS. See module docstring for the rule.

    `visited` is the cycle guard for THIS run; `discovered` is the accumulating
    output manifest (pre-loaded on re-runs). Keeping them separate means a re-run
    actually re-walks pages already in the manifest — so an improved link filter
    can surface newly-reachable children — instead of short-circuiting on the very
    first page because it was discovered in a prior run.
    """
    if depth > MAX_DEPTH:
        return
    if depth > 0:
        if url in visited:
            return
        if limit_target is not None and len(discovered) >= limit_target:
            return

    try:
        # depth 0 root was already announced by the caller — suppress its fetch line
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
        # Tree root — accept unconditionally, use its breadcrumb for children.
        category = infer_category(current_crumb, base_category)
        discovered[url] = category
        visited.add(url)
        crumb_for_children = current_crumb
    else:
        if not current_crumb:
            _note_no_breadcrumb(url, no_crumb)
            if verbose:
                console.print(f"  [dim][no-breadcrumb, logged][/dim] {url}")
            return

        # One-level descent rule
        if len(current_crumb) != len(parent_crumb) + 1:
            return
        if current_crumb[:len(parent_crumb)] != parent_crumb:
            return

        category = infer_category(current_crumb, base_category)
        discovered[url] = category
        visited.add(url)
        crumb_for_children = current_crumb

        if verbose:
            console.print(
                f"  [#{len(discovered):04d}] [{category}] {url.split('/')[-1]}  "
                f"[dim]{' / '.join(current_crumb)}[/dim]"
            )
        if limit_target is not None and len(discovered) >= limit_target:
            return

    # Periodic progress milestone (every 100 pages), regardless of verbosity.
    # Shows the running count and the breadcrumb path currently being explored.
    if len(discovered) >= progress["next_milestone"]:
        path = " / ".join(crumb_for_children) if crumb_for_children else "(root)"
        console.print(
            f"  [bold green]{len(discovered)} pages discovered[/bold green] "
            f"[dim]— exploring: {path}[/dim]"
        )
        # Advance to the next multiple of 100 above the current count
        progress["next_milestone"] = ((len(discovered) // 100) + 1) * 100

    # Recurse into content links (footer nav tables already excluded above)
    for child_url in extract_links_for_discovery(BeautifulSoup(html, "lxml")):
        if child_url not in visited:
            await crawl(
                client=client, sem=sem, last_t=last_t,
                url=child_url, parent_crumb=crumb_for_children,
                base_category=base_category, discovered=discovered,
                visited=visited, no_crumb=no_crumb, progress=progress,
                limit_target=limit_target, verbose=verbose,
                depth=depth + 1,
            )


# ── Persistence helpers (Option A: append + dedupe) ──────────────────────────

def _load_existing_manifest() -> dict[str, str]:
    """Load any existing discovered_urls.jsonl so re-runs merge instead of overwrite."""
    out: dict[str, str] = {}
    if not DISCOVERED_FILE.exists():
        return out
    with DISCOVERED_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                out[obj["url"]] = obj["category"]
            except Exception:
                pass
    return out


def _load_existing_no_crumb() -> set[str]:
    """Load any existing no_breadcrumb.txt so re-runs merge instead of overwrite."""
    out: set[str] = set()
    if not NO_BREADCRUMB_FILE.exists():
        return out
    with NO_BREADCRUMB_FILE.open(encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u:
                out.add(u)
    return out


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def discover(
    entry:   Optional[str] = None,
    limit:   Optional[int] = None,
    verbose: bool = False,
    dry_run: bool = False,
) -> dict[str, str]:
    # Option A: pre-load existing results so running entry points one at a time
    # accumulates into a single manifest. Re-running an entry refreshes its own
    # URLs (keyed by URL) without disturbing the others.
    discovered: dict[str, str] = _load_existing_manifest()
    no_crumb: set[str] = _load_existing_no_crumb()
    # Per-run cycle guard, deliberately NOT pre-loaded from the manifest: on a
    # re-run we want to re-walk known pages (the manifest only dedups the OUTPUT),
    # so an improved link filter can surface newly-reachable children.
    visited: set[str] = set()

    if discovered:
        console.print(
            f"[dim]Loaded {len(discovered)} existing URLs from "
            f"{DISCOVERED_FILE} — new results will be merged in.[/dim]"
        )

    entries = ENTRY_POINTS
    if entry:
        entries = [(p, c) for p, c in ENTRY_POINTS if p == entry]
        if not entries:
            console.print(f"[red]Unknown entry point: {entry}[/red]")
            console.print(f"Valid options: {[p for p, _ in ENTRY_POINTS]}")
            return {}

    sem    = asyncio.Semaphore(1)
    last_t = [0.0]
    # Start the milestone counter above whatever we already had loaded
    progress = {"next_milestone": ((len(discovered) // 100) + 1) * 100}

    # --limit means 'N NEW pages this run', so the absolute stop target is
    # the pre-loaded baseline plus the requested limit.
    baseline = len(discovered)
    limit_target = (baseline + limit) if limit is not None else None

    async with build_client() as client:
        for path, base_category in entries:
            entry_url = BASE_URL + path
            console.print(
                f"\n[bold cyan]Entry point:[/bold cyan] {path} "
                f"(base category: {base_category})"
            )
            before = len(discovered)
            await crawl(
                client=client, sem=sem, last_t=last_t,
                url=entry_url, parent_crumb=[],
                base_category=base_category, discovered=discovered,
                visited=visited, no_crumb=no_crumb, progress=progress,
                limit_target=limit_target, verbose=verbose, depth=0,
            )
            console.print(f"  → {len(discovered) - before} new pages under {path}")
            if limit_target is not None and len(discovered) >= limit_target:
                break

    # Summary
    console.print(
        f"\n[bold green]Discovery complete:[/bold green] {len(discovered)} URLs"
    )
    if no_crumb:
        console.print(
            f"[yellow]{len(no_crumb)} pages had no breadcrumb[/yellow] "
            f"→ {NO_BREADCRUMB_FILE}"
        )
    by_cat: dict[str, int] = defaultdict(int)
    for cat in discovered.values():
        by_cat[cat] += 1
    for cat, n in sorted(by_cat.items(), key=lambda x: -x[1]):
        console.print(f"  {cat:12s} {n:5d}")

    if dry_run:
        console.print("\n[yellow]Dry run — nothing written.[/yellow]")
        return discovered

    # Persist the merged manifest (Option A: contains prior + new, deduped by URL)
    DISCOVERED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DISCOVERED_FILE.open("w", encoding="utf-8") as f:
        for url, category in discovered.items():
            f.write(json.dumps({"url": url, "category": category}, ensure_ascii=False) + "\n")
    console.print(f"\n[bold green]Saved → {DISCOVERED_FILE}[/bold green] "
                  f"({len(discovered)} URLs total)")

    # Rewrite the merged no-breadcrumb list (prior + new, deduped)
    if no_crumb:
        with NO_BREADCRUMB_FILE.open("w", encoding="utf-8") as f:
            for u in sorted(no_crumb):
                f.write(u + "\n")
        console.print(f"[dim]No-breadcrumb list → {NO_BREADCRUMB_FILE} "
                      f"({len(no_crumb)} URLs)[/dim]")

    return discovered


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elden Ring wiki — discovery phase (breadcrumb DFS)"
    )
    parser.add_argument("--entry", type=str, default=None,
                        help="Restrict to one entry point, e.g. /World+Information")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after discovering N pages (testing)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each page as it is discovered")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover and summarise, but do not write the manifest")
    args = parser.parse_args()
    asyncio.run(discover(
        entry=args.entry, limit=args.limit,
        verbose=args.verbose, dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()