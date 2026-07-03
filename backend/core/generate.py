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
    def stream(self, result: PipelineResult,
               history: Optional[str] = None) -> Iterator[str]:
        """
        Yield answer tokens for a PipelineResult. The future SSE endpoint wraps
        this directly.
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
            for chunk in self._llm.stream(messages):
                text = getattr(chunk, "content", "")
                if text:
                    yield text
            return

        # Retrieved but empty → don't invoke the LLM on nothing; give guidance.
        if not result.chunks:
            yield _NO_CONTEXT_FALLBACK
            return

        messages = build_messages(
            question=result.question,
            chunks=result.chunks,
            tone=tone,
            history=history,
        )
        for chunk in self._llm.stream(messages):
            text = getattr(chunk, "content", "")
            if text:
                yield text

    # ── metadata (sources + images) ──────────────────────────────────────────
    @staticmethod
    def assemble_metadata(result: PipelineResult) -> tuple[list[dict], list[dict]]:
        sources = format_sources(result.chunks) if result.chunks else []
        images: list[dict] = []
        if result.chunks:
            images = select_images(
                result.chunks,
                entity_focus=result.decision.entity_focus,
            )
        return sources, images


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
        sources, images = Generator.assemble_metadata(result)
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
    console.print(f"[dim]Loading assistant…[/dim]")
    assistant = Assistant()

    console.print(f"\n[bold cyan]Q:[/bold cyan] {question}\n")
    console.print("[bold]Answer:[/bold]")
    # Stream to the console live
    result = assistant._pipeline.run(question, k=args.k)
    console.print(f"[dim](tone={result.decision.tone}, "
                  f"chunks={len(result.chunks)}, "
                  f"fallback={result.entity_fallback})[/dim]\n")
    for token in assistant._generator.stream(result):
        console.print(token, end="")
    console.print()

    sources, images = Generator.assemble_metadata(result)
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