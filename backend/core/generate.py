"""
backend/core/generate.py

The generation layer — turns retrieved chunks into a streamed, tone-aware,
grounded answer. This is the last core component: with it, the full RAG stack
answers a question end to end.

Design:
  - Generator is PURE: it takes a PipelineResult and streams answer tokens. This
    is the building block the future /chat SSE endpoint will use, calling
    pipeline and generation as separate stages so it can stream + handle errors
    per stage.
  - answer_question() is a CONVENIENCE full-stack runner (question → pipeline →
    generate) for testing and simple callers.

The tone (scholar/cryptic) comes straight from the router's decision, so the
voice is automatic from the user's question. Sources and the optional entity
image are assembled from the chunks for the UI to render as cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

from core.pipeline import Pipeline, PipelineResult
from core.prompts import build_messages, format_sources, select_images
from core.llm import get_chat_llm

# A friendly message when there's nothing to answer from.
_NO_CONTEXT_FALLBACK = (
    "I couldn't find anything in the wiki about that. Try naming a specific "
    "boss, item, location, or NPC — for example, “where do I find the "
    "Meteorite Staff?” or “how do I beat Margit?”"
)


def _chunk_text(chunk) -> str:
    """Normalize a streamed chunk's content to a plain string. Gemini/Gemma can
    return `content` as a str OR a list of content blocks (str or {'type':'text',
    'text':...}); the list form broke '"".join(parts)' downstream. Always coerce."""
    c = getattr(chunk, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for b in c:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                out.append(b.get("text") or b.get("content") or "")
        return "".join(out)
    return str(c) if c else ""


def _extract_model(chunk) -> Optional[str]:
    """Best-effort: which model produced this streamed chunk (after any fallback).
    Provider metadata varies, so check the usual spots and take the first hit."""
    for src in (getattr(chunk, "response_metadata", None),
                getattr(chunk, "additional_kwargs", None)):
        if isinstance(src, dict):
            for key in ("model_name", "model"):
                v = src.get(key)
                if v:
                    return str(v)
    return None


@dataclass
class GenerationResult:
    """Everything the UI needs: the answer text, sources, images, and metadata."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)   # [{image_url, title, url}]
    tone: str = "scholar"
    used_retrieval: bool = True
    entity_fallback: bool = False


class Generator:
    def __init__(self) -> None:
        self._llm = get_chat_llm()

    # ── streaming (pure) ─────────────────────────────────────────────────────
    def stream(self, result: PipelineResult, history: Optional[str] = None,
               trace: Optional[dict] = None) -> Iterator[str]:
        """
        Yield answer tokens for a PipelineResult. The SSE endpoints wrap this
        directly. Pass a mutable `trace` dict to capture which model actually
        answered (after any fallback) as trace["model"] — read it once the
        generator is exhausted.
        """
        decision = result.decision
        tone = decision.tone

        # No retrieval (greeting/chitchat) → answer conversationally, no context.
        if not result.retrieved:
            messages = [
                ("system",
                 "You are a friendly Elden Ring assistant. The player said "
                 "something conversational, not a wiki question. Reply briefly "
                 "and warmly, and invite them to ask about the game."),
                ("human", result.question),
            ]
            yield from self._stream_llm(messages, trace)
            return

        # Retrieved but empty → don't invoke the LLM on nothing; give guidance.
        if not result.chunks:
            if trace is not None:
                trace["model"] = "none:no-context-fallback"
            yield _NO_CONTEXT_FALLBACK
            return

        messages = build_messages(
            question=result.question,
            chunks=result.chunks,
            tone=tone,
            history=history,
            enumerate_group=decision.enumerate_group,
            roster=result.roster,
        )
        yield from self._stream_llm(messages, trace)

    def _stream_llm(self, messages, trace: Optional[dict]) -> Iterator[str]:
        """Stream tokens from the chat chain, recording the answering model."""
        for chunk in self._llm.stream(messages):
            if trace is not None and "model" not in trace:
                m = _extract_model(chunk)
                if m:
                    trace["model"] = m
            text = _chunk_text(chunk)
            if text:
                yield text

    # ── metadata (sources + images) ──────────────────────────────────────────
    @staticmethod
    def assemble_metadata(
        result: PipelineResult, answer_text: Optional[str] = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Build source cards + images for the UI. Pass the generated `answer_text`
        so media is gated to what the answer actually references (drops tangential
        images/sources and clears them on refusals).
        """
        if not result.chunks:
            return [], []
        sources = format_sources(
            result.chunks,
            entity_focus=result.decision.entity_focus,
            answer_text=answer_text,
        )
        images = select_images(
            result.chunks,
            entity_focus=result.decision.entity_focus,
            answer_text=answer_text,
        )
        return sources, images


def answer_stats(result: PipelineResult, model: Optional[str],
                 extra_timings: dict) -> dict:
    """
    One flat dict summarising a request for structured logging + the SSE `done`
    event: which model answered, routing decision, retrieval shape, and the full
    timing breakdown (pipeline's router_ms/retrieval_ms merged with the caller's
    ttft_ms/gen_ms/total_ms).
    """
    d = result.decision
    top = [
        {"title": c.title, "score": round(c.final_score if c.boost else c.score, 3)}
        for c in result.chunks[:3]
    ]
    return {
        "model": model,
        "intent": d.intent,
        "tone": d.tone,
        "entity_focus": d.entity_focus,
        "entity_fallback": result.entity_fallback,
        "used_retrieval": result.retrieved,
        "chunk_count": len(result.chunks),
        "enumerate_group": d.enumerate_group,
        "roster_count": len(result.roster),
        "top_chunks": top,
        "timings": {**result.timings, **extra_timings},
    }


# ── Full-stack convenience runner ─────────────────────────────────────────────

class Assistant:
    """question → pipeline → generation, in one object. For testing/simple use."""

    def __init__(self) -> None:
        self._pipeline = Pipeline()
        self._generator = Generator()

    def stream_answer(self, question: str, history: Optional[str] = None,
                      k: int = 8) -> Iterator[str]:
        result = self._pipeline.run(question, history=history, k=k)
        yield from self._generator.stream(result, history=history)

    def answer(self, question: str, history: Optional[str] = None,
               k: int = 8) -> GenerationResult:
        """Non-streaming convenience: collects the stream and bundles metadata."""
        result = self._pipeline.run(question, history=history, k=k)
        text = "".join(self._generator.stream(result, history=history))
        sources, images = Generator.assemble_metadata(result, answer_text=text)
        return GenerationResult(
            answer=text,
            sources=sources,
            images=images,
            tone=result.decision.tone,
            used_retrieval=result.retrieved,
            entity_fallback=result.entity_fallback,
        )


# ── CLI: type a question, watch it stream ─────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Ask the Elden Ring assistant")
    p.add_argument("question", nargs="*", help="Your question")
    p.add_argument("-k", type=int, default=8)
    args = p.parse_args()

    from rich.console import Console
    console = Console()

    question = " ".join(args.question) or "Tell me the story of Queen Marika"
    console.print("[dim]Loading assistant…[/dim]")
    assistant = Assistant()

    console.print(f"\n[bold cyan]Q:[/bold cyan] {question}\n")
    console.print("[bold]Answer:[/bold]")
    # Stream to the console live
    result = assistant._pipeline.run(question, k=args.k)
    console.print(f"[dim](tone={result.decision.tone}, "
                  f"chunks={len(result.chunks)}, "
                  f"fallback={result.entity_fallback})[/dim]\n")
    parts: list[str] = []
    for token in assistant._generator.stream(result):
        parts.append(token)
        console.print(token, end="")
    console.print()

    sources, images = Generator.assemble_metadata(result, answer_text="".join(parts))
    if sources:
        console.print("\n[bold]Sources:[/bold]")
        for s in sources[:5]:
            console.print(f"  • {s['title']} — {s['url']}")
    if images:
        console.print(f"\n[bold]Images ({len(images)}):[/bold]")
        for im in images:
            console.print(f"  • {im['title']}: {im['image_url']}")


if __name__ == "__main__":
    main()