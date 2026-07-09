"""
backend/eval/bench.py

Systematic RAG evaluation harness for Elden Path.

Runs a benchmark question set through the full stack (Router → Pipeline →
Generator) and produces a JSON data file + Markdown summary report.

Usage (from backend/):
    uv run python eval/bench.py                # auto-scores, all questions
    uv run python eval/bench.py --judge        # also use Gemini as LLM judge
    uv run python eval/bench.py --no-gen       # routing + retrieval only (faster)
    uv run python eval/bench.py -q "question"  # single ad-hoc question

Outputs:
    eval/results/bench_YYYYMMDD_HHMMSS.json
    eval/results/bench_YYYYMMDD_HHMMSS.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make 'backend/' the importable root regardless of CWD.
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.pipeline import Pipeline
from core.generate import Generator


# ── Scoring ───────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    # Strip commas, normalize apostrophe variants (curly vs straight), and
    # collapse whitespace so e.g. "Radagon’s Soreseal" matches "Radagon's Soreseal".
    s = s.lower().replace(",", " ").replace("’", "'").replace("‘", "'")
    return " ".join(s.split())


def auto_score(q: dict, result, chunks: list, answer: str) -> dict:
    """Heuristic scores that require no LLM call."""
    exp_entity = q.get("expected_entity", "")
    exp_intent = q.get("expected_intent", "")
    entity_focus = result.decision.entity_focus

    entity_hit: Optional[bool] = None
    if exp_entity:
        entity_hit = any(
            _norm(exp_entity) in _norm(e) or _norm(e) in _norm(exp_entity)
            for e in entity_focus
        )

    intent_hit: Optional[bool] = None
    if exp_intent:
        intent_hit = result.decision.intent == exp_intent

    # Enumeration: did the router set the right category AND did the metadata roster
    # actually populate? (validates the "how many / list all X" path end-to-end)
    exp_enum = q.get("expected_enumerate_group", "")
    enumerate_hit: Optional[bool] = None
    if exp_enum:
        got_enum = result.decision.enumerate_group or ""
        enumerate_hit = (_norm(exp_enum) == _norm(got_enum)
                         and len(getattr(result, "roster", [])) > 0)

    top1_title_match: Optional[bool] = None
    if chunks and exp_entity:
        top1_title_match = _norm(exp_entity) in _norm(chunks[0].title)

    top3_coverage: Optional[float] = None
    if chunks and exp_entity:
        hits = sum(1 for c in chunks[:3] if _norm(exp_entity) in _norm(c.title))
        top3_coverage = round(hits / min(3, len(chunks)), 2)

    answer_mentions: Optional[bool] = None
    if answer and exp_entity:
        answer_mentions = _norm(exp_entity) in _norm(answer)

    return {
        "entity_hit": entity_hit,
        "intent_hit": intent_hit,
        "enumerate_hit": enumerate_hit,
        "top1_title_match": top1_title_match,
        "top3_entity_coverage": top3_coverage,
        "entity_fallback_used": result.entity_fallback,
        "answer_mentions_entity": answer_mentions,
        "chunks_returned": len(chunks),
    }


def llm_judge(question: str, answer: str, llm) -> dict:
    """Call the configured LLM to score an answer on a rubric."""
    if not answer.strip():
        return {"grounded": 0, "complete": 0, "concise": 0, "flag": "empty answer"}
    prompt = (
        "You are evaluating an Elden Ring RAG chatbot answer. "
        "Score it on three dimensions, 1-5 each. Reply with ONLY valid JSON.\n\n"
        f"Question: {question}\n\nAnswer: {answer}\n\n"
        "Scoring:\n"
        "- grounded: 5=only verifiable Elden Ring facts, 1=obvious hallucination\n"
        "- complete: 5=fully answers the question, 1=dodges or entirely irrelevant\n"
        "- concise: 5=tight prose no filler, 1=bloated repetitive hedging\n"
        "- flag: one short phrase for the main issue, or \"ok\"\n\n"
        'Return only: {"grounded": N, "complete": N, "concise": N, "flag": "..."}'
    )
    try:
        resp = llm.invoke([("human", prompt)])
        text = resp.content.strip()
        start, end = text.find("{"), text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception as exc:
        return {"grounded": -1, "complete": -1, "concise": -1, "flag": f"judge error: {exc}"}
    return {"grounded": -1, "complete": -1, "concise": -1, "flag": "parse error"}


# ── Per-question run ──────────────────────────────────────────────────────────

_ERROR_SCORES: dict = {
    "entity_hit": None, "intent_hit": None, "enumerate_hit": None, "top1_title_match": None,
    "top3_entity_coverage": None, "entity_fallback_used": False,
    "answer_mentions_entity": None, "chunks_returned": 0,
}


def run_question(q: dict, pipeline: Pipeline, generator: Generator,
                 skip_gen: bool = False, max_retries: int = 3) -> dict:
    base = {
        "id": q.get("id", q["question"][:40]),
        "question": q["question"],
        "category": q.get("category", "unknown"),
        "expected_entity": q.get("expected_entity", ""),
        "expected_intent": q.get("expected_intent", ""),
        "notes": q.get("notes", ""),
    }
    for attempt in range(max_retries + 1):
        try:
            return _run_question_inner(q, pipeline, generator, skip_gen, base)
        except Exception as exc:
            msg = str(exc)
            is_rate_limit = "429" in msg or "too many requests" in msg.lower()
            if is_rate_limit and attempt < max_retries:
                wait = 60 * (attempt + 1)
                print(f"  429 rate limit — waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
                continue
            print(f"  ERROR: {type(exc).__name__}: {msg[:120]}")
            return {
                **base,
                "error": f"{type(exc).__name__}: {msg[:300]}",
                "routing": {}, "retrieval": [],
                "generation": {"answer": "", "answer_len": 0, "sources": [], "images": []},
                "timing": {"route_ms": -1, "gen_ms": -1},
                "auto_scores": _ERROR_SCORES,
            }


def _run_question_inner(q: dict, pipeline: Pipeline, generator: Generator,
                        skip_gen: bool, base: dict) -> dict:
    t0 = time.perf_counter()
    result = pipeline.run(q["question"])
    route_ms = int((time.perf_counter() - t0) * 1000)

    chunks = result.chunks
    answer, sources, images, gen_ms = "", [], [], 0

    if result.chunks:
        # Always collect sources + images — these depend only on retrieved chunks,
        # not on the generation LLM, so they're available even with --no-gen.
        sources, images = Generator.assemble_metadata(result)

    if not skip_gen:
        t1 = time.perf_counter()
        answer = "".join(generator.stream(result))
        gen_ms = int((time.perf_counter() - t1) * 1000)

    d = result.decision
    return {
        **base,
        "routing": {
            "entity_focus": d.entity_focus,
            "raw_mentions": d.raw_mentions,
            "intent": d.intent,
            "tone": d.tone,
            "category_hint": d.category_hint,
            "chunk_type_bias": d.chunk_type_bias,
            "needs_image": d.needs_image,
            "used_call2": d.used_call2,
            "entity_fallback": result.entity_fallback,
            "resolution_debug": d.resolution_debug,
        },
        "retrieval": [
            {
                "rank": i + 1,
                "title": c.title,
                "section": c.section_heading,
                "category": c.category,
                "chunk_type": c.chunk_type,
                "score": round(c.score, 4),
                "final_score": round(c.final_score, 4) if c.boost else None,
                "has_image": bool(c.image_url),
                "entities_preview": c.entities[:3] if hasattr(c, "entities") else [],
            }
            for i, c in enumerate(chunks)
        ],
        "generation": {
            "answer": answer,
            "answer_len": len(answer),
            "sources": [s["title"] for s in sources],
            "images": [im["title"] for im in images],
        },
        "timing": {"route_ms": route_ms, "gen_ms": gen_ms},
        "auto_scores": auto_score(q, result, chunks, answer),
    }


# ── Markdown report ───────────────────────────────────────────────────────────

def _fmt_bool(v: Optional[bool]) -> str:
    if v is None:
        return "—"
    return "✓" if v else "✗"


def build_markdown(results: list[dict], judge_mode: bool) -> str:
    lines = ["# Elden Path — Benchmark Results\n"]

    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    # Overall summary
    all_entity = [r["auto_scores"]["entity_hit"] for r in results if r["auto_scores"]["entity_hit"] is not None]
    all_intent = [r["auto_scores"]["intent_hit"] for r in results if r["auto_scores"]["intent_hit"] is not None]
    all_enum   = [r["auto_scores"].get("enumerate_hit") for r in results if r["auto_scores"].get("enumerate_hit") is not None]
    all_top1   = [r["auto_scores"]["top1_title_match"] for r in results if r["auto_scores"]["top1_title_match"] is not None]
    _enum = f" | enumerate {sum(all_enum)}/{len(all_enum)}" if all_enum else ""
    lines.append(
        f"**Overall** — entity_hit {sum(all_entity)}/{len(all_entity)} "
        f"| intent_hit {sum(all_intent)}/{len(all_intent)} "
        f"| top1_match {sum(all_top1)}/{len(all_top1)}{_enum}\n"
    )

    for cat, rs in sorted(by_cat.items()):
        lines.append(f"\n## {cat.upper()}\n")

        judge_hdr = " | G | C | Cn | Flag" if judge_mode else ""
        lines.append(f"| ID | Question | Intent | Ent✓ | Top1 | Top3 | Fallback | AnsLen{judge_hdr} |")
        lines.append(f"|---|---|---|---|---|---|---|---{'|---|---|---|---|' if judge_mode else '|'}")

        for r in rs:
            s = r["auto_scores"]
            j = r.get("judge", {})
            intent = r["routing"].get("intent") or "--"
            intent_ok = _fmt_bool(s["intent_hit"])
            top3 = f"{s['top3_entity_coverage']:.0%}" if s["top3_entity_coverage"] is not None else "--"
            q_short = r["question"][:50].replace("|", "/")
            ans_len = r["generation"]["answer_len"]
            row = (
                f"| `{r['id']}` | {q_short} | {intent} {intent_ok} "
                f"| {_fmt_bool(s['entity_hit'])} | {_fmt_bool(s['top1_title_match'])} "
                f"| {top3} | {_fmt_bool(s['entity_fallback_used'])} | {ans_len}"
            )
            if judge_mode and j:
                row += f" | {j.get('grounded','?')} | {j.get('complete','?')} | {j.get('concise','?')} | {j.get('flag','')}"
            lines.append(row + " |")

        # Category aggregate
        e_hits = [r["auto_scores"]["entity_hit"] for r in rs if r["auto_scores"]["entity_hit"] is not None]
        i_hits = [r["auto_scores"]["intent_hit"] for r in rs if r["auto_scores"]["intent_hit"] is not None]
        t_hits = [r["auto_scores"]["top1_title_match"] for r in rs if r["auto_scores"]["top1_title_match"] is not None]
        lines.append(
            f"\n> **{cat}**: entity {sum(e_hits)}/{len(e_hits)} "
            f"| intent {sum(i_hits)}/{len(i_hits)} "
            f"| top1 {sum(t_hits)}/{len(t_hits)}"
        )

    # Failures section
    lines.append("\n\n## Failures & Flags\n")
    failures = []
    for r in results:
        s = r["auto_scores"]
        j = r.get("judge", {})
        issues = []
        if s["entity_hit"] is False:
            issues.append(f"entity missed (got focus={r['routing'].get('entity_focus', [])})")
        if s["intent_hit"] is False:
            issues.append(f"intent wrong (got={r['routing'].get('intent', '?')}, expected={r['expected_intent']})")
        if s["top1_title_match"] is False and r["retrieval"]:
            issues.append(f"top chunk = '{r['retrieval'][0]['title']}'")
        if s["entity_fallback_used"]:
            issues.append("entity filter empty → fallback")
        if judge_mode and j.get("complete", 5) <= 2:
            issues.append(f"judge: incomplete ({j.get('flag', '')})")
        if r.get("notes"):
            issues.append(f"note: {r['notes']}")
        if issues:
            failures.append(f"- **{r['id']}** `{r['question'][:55]}`: {'; '.join(issues)}")
    lines.extend(failures if failures else ["_No failures detected._"])

    # Retrieval detail
    lines.append("\n\n## Retrieval Detail (top-3 chunks per question)\n")
    for r in results:
        lines.append(f"\n**{r['id']}**: _{r['question']}_")
        if not r["retrieval"]:
            lines.append("  _(no retrieval — chitchat or no_retrieval category)_")
            continue
        for c in r["retrieval"][:3]:
            fs = f" → final={c['final_score']:.3f}" if c["final_score"] else ""
            lines.append(
                f"  {c['rank']}. `{c['title']} · {c['section']}` "
                f"score={c['score']:.3f}{fs} [{c['chunk_type']}]"
            )

    # Full answers section
    lines.append("\n\n## Generated Answers\n")
    for r in results:
        ans = r["generation"]["answer"]
        if not ans:
            continue
        lines.append(f"\n### {r['id']}: {r['question']}\n")
        lines.append(ans.strip())
        if r["generation"]["sources"]:
            lines.append(f"\n_Sources: {', '.join(r['generation']['sources'][:4])}_")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Elden Path RAG benchmark")
    p.add_argument("--questions", default=str(Path(__file__).parent / "questions.json"),
                   help="Path to benchmark questions JSON")
    p.add_argument("--judge", action="store_true",
                   help="Use the configured LLM as judge for answer quality")
    p.add_argument("--no-gen", action="store_true",
                   help="Skip generation (routing + retrieval diagnostics only, much faster)")
    p.add_argument("-q", "--single", metavar="QUESTION",
                   help="Run a single ad-hoc question instead of the full set")
    p.add_argument("-k", type=int, default=8, help="Number of chunks to retrieve")
    p.add_argument("--delay", type=float, default=3.0,
                   help="Seconds to sleep between questions (avoids NIM rate limits, default 3)")
    args = p.parse_args()

    from rich.console import Console
    # force_terminal=False avoids the Windows legacy console encoder that
    # chokes on non-CP1252 characters (checkmarks, dashes, etc.)
    console = Console(force_terminal=False, highlight=False)

    print("Elden Path - RAG Benchmark")
    print("Initialising pipeline (LLM + embedder + Qdrant)...")

    pipeline = Pipeline()
    generator = Generator()

    judge_fn = None
    if args.judge:
        from core.llm import get_chat_llm
        _judge_llm = get_chat_llm()
        judge_fn = lambda q, a: llm_judge(q, a, _judge_llm)
        print("LLM judge enabled")

    if args.single:
        questions = [{"id": "adhoc", "question": args.single, "category": "adhoc",
                      "expected_entity": "", "expected_intent": ""}]
    else:
        with open(args.questions, encoding="utf-8") as f:
            questions = json.load(f)

    print(f"Running {len(questions)} question(s)...\n")

    results = []
    for idx, q in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] {q['question'][:65]}")
        r = run_question(q, pipeline, generator, skip_gen=args.no_gen)

        if judge_fn and r["generation"]["answer"]:
            r["judge"] = judge_fn(q["question"], r["generation"]["answer"])

        results.append(r)

        if idx < len(questions) and args.delay > 0:
            time.sleep(args.delay)

        # Inline status (ASCII only — Windows console CP1252 compatibility)
        s = r["auto_scores"]
        bits = []
        if s["entity_hit"] is True:    bits.append("entity:OK")
        elif s["entity_hit"] is False: bits.append("entity:FAIL")
        if s["intent_hit"] is True:    bits.append("intent:OK")
        elif s["intent_hit"] is False: bits.append("intent:FAIL")
        if s.get("enumerate_hit") is True:    bits.append("enum:OK")
        elif s.get("enumerate_hit") is False: bits.append("enum:FAIL")
        if s["top1_title_match"] is True:    bits.append("top1:OK")
        elif s["top1_title_match"] is False: bits.append("top1:FAIL")
        if s["entity_fallback_used"]: bits.append("FALLBACK")
        print("  " + (" ".join(bits) if bits else "--"))

    # Write outputs — JSON first so data is saved even if markdown render fails
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"bench_{ts}.json"
    md_path   = out_dir / f"bench_{ts}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote: {json_path}")

    md = build_markdown(results, judge_mode=bool(judge_fn))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
