"""
eval/run_conversations.py

Smoke-runner for multi-turn / memory behaviour — complements bench.py (which
scores single-turn routing+retrieval). Reads a plain-text question list where a
line may hold a follow-up after a pipe:

    How do I reach Rykard? | What does he drop (follow up question)

and runs it as a REAL conversation: answer turn 1 → feed it back as history →
run turn 2. This is what exercises the router's follow-up rewrite + the
affirmation guard + the generator's CONFIRMATIONS rule. It also appends a couple
of built-in "Yes"-after-offer conversations to regression-guard the confirmation
bug (a bare "Yes" must continue the thread, not fall back to "ask me a question").

Output is Q/A prose for eyeballing — there are no gold labels here (the txt file
has none); reliability is judged by reading, not scored.

Usage (from backend/):
    uv run python -m eval.run_conversations                       # all lines
    uv run python -m eval.run_conversations --only-followups      # just the |-lines + confirmations
    uv run python -m eval.run_conversations --limit 10 --delay 5  # a subset, gentle on rate limits
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

from core.pipeline import Pipeline
from core.generate import Generator

DEFAULT_TXT = Path(__file__).parent / "eval_questions.txt"
RESULTS_DIR = Path(__file__).parent / "results"

# Built-in confirmation conversations — the exact class that broke on the webpage.
# Turn 1 elicits an overview + an offer; the bare turn-2 reply must continue it.
_CONFIRMATION_CONVOS = [
    ["How do I progress Ranni's questline?", "Yes"],
    ["Give me an overview of Fia's questline", "yes please, the full guide"],
]

_FOLLOWUP_TAG = re.compile(r"\s*\(follow[\s-]*up(?:\s+question)?\)\s*$", re.I)


def parse_line(line: str) -> list[str] | None:
    """A txt line → list of turns. '|' splits turns; a trailing '(follow up …)'
    annotation is stripped. Returns None for blank lines."""
    line = line.strip()
    if not line:
        return None
    turns = [t.strip() for t in line.split("|")]
    turns = [_FOLLOWUP_TAG.sub("", t).strip() for t in turns]
    return [t for t in turns if t]


def _route_summary(d) -> str:
    return (f"needs_retrieval={d.needs_retrieval} intent={d.intent} "
            f"entity={d.entity_focus} enumerate={d.enumerate_group} "
            f"rq={d.retrieval_query!r}")


def run_convo(turns: list[str], pipe: Pipeline, gen: Generator,
              delay: float) -> list[dict]:
    """Run one conversation; return a per-turn record with routing + answer."""
    history_parts: list[str] = []
    records = []
    for i, q in enumerate(turns):
        history = "\n".join(history_parts) if history_parts else None
        res = pipe.run(q, history=history)
        answer = "".join(gen.stream(res, history=history))
        records.append({
            "turn": i + 1, "q": q,
            "route": _route_summary(res.decision),
            "chunks": len(res.chunks),
            "answer": answer,
        })
        history_parts.append(f"User: {q}")
        history_parts.append(f"Assistant: {answer}")
        if delay > 0 and i < len(turns) - 1:
            time.sleep(delay)
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multi-turn eval conversations")
    ap.add_argument("--txt", default=str(DEFAULT_TXT))
    ap.add_argument("--only-followups", action="store_true",
                    help="Only run lines with a '|' follow-up, plus confirmations")
    ap.add_argument("--limit", type=int, default=0, help="Cap number of conversations")
    ap.add_argument("--start", type=int, default=0, help="Skip the first N conversations")
    ap.add_argument("--delay", type=float, default=5.0, help="Seconds between LLM turns (rate limit)")
    ap.add_argument("--no-confirmations", action="store_true",
                    help="Skip the built-in Yes-after-offer conversations")
    args = ap.parse_args()

    from rich.console import Console
    console = Console()

    lines = Path(args.txt).read_text(encoding="utf-8").splitlines()
    convos: list[list[str]] = []
    for ln in lines:
        turns = parse_line(ln)
        if not turns:
            continue
        if args.only_followups and len(turns) < 2:
            continue
        convos.append(turns)
    if not args.no_confirmations:
        convos.extend(_CONFIRMATION_CONVOS)

    convos = convos[args.start:]
    if args.limit:
        convos = convos[: args.limit]

    console.print("[dim]Loading pipeline...[/dim]")
    pipe, gen = Pipeline(), Generator()

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"conversations_{datetime.now():%Y%m%d_%H%M%S}.md"
    blocks: list[str] = []

    for n, turns in enumerate(convos, 1):
        tag = "multi-turn" if len(turns) > 1 else "single"
        console.print(f"[cyan]\\[{n}/{len(convos)}][/cyan] ({tag}) {turns[0][:60]}...")
        try:
            records = run_convo(turns, pipe, gen, args.delay)
        except Exception as e:  # a rate-limit/tier failure on one convo shouldn't kill the run
            console.print(f"  [red]ERROR: {type(e).__name__}: {e}[/red]")
            blocks.append(f"## {n}. ({tag}) — ERROR\n**T1 Q:** {turns[0]}\n\n`{type(e).__name__}: {e}`")
            if args.delay > 0 and n < len(convos):
                time.sleep(args.delay)
            continue
        lines_md = [f"## {n}. ({tag})"]
        for r in records:
            lines_md.append(f"**T{r['turn']} Q:** {r['q']}")
            lines_md.append(f"`{r['route']}` - chunks={r['chunks']}")
            lines_md.append(f"**A:** {r['answer']}\n")
            # console preview (ASCII-only so a cp1252 console can't crash the run)
            preview = r["answer"][:160].strip().encode("ascii", "replace").decode()
            console.print(f"  [dim]T{r['turn']}[/dim] {r['route']}")
            console.print(f"       {preview}...")
        blocks.append("\n".join(lines_md))
        # Write after every convo so a long run's partial output survives interruption.
        out_path.write_text("# Multi-turn eval run\n\n" + "\n\n---\n\n".join(blocks),
                            encoding="utf-8")
        if args.delay > 0 and n < len(convos):
            time.sleep(args.delay)

    out_path.write_text("# Multi-turn eval run\n\n" + "\n\n---\n\n".join(blocks),
                        encoding="utf-8")
    console.print(f"\n[green]Wrote {len(convos)} conversations -> {out_path}[/green]")


if __name__ == "__main__":
    main()
