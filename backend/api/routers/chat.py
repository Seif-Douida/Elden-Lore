"""
backend/api/routers/chat.py

The /chat endpoint — streams a grounded, tone-aware answer over SSE.

Event protocol (event: <type>\\ndata: <json>\\n\\n):
    meta     once, first   {tone, used_retrieval, entity_fallback}
    token    many          {text}
    sources  once          {sources: [...]}
    images   once          {images: [...]}
    done     once, last    {}
    error    on failure    {message}

Inline image placement is a planned second pass; this sends images as a final
event.
"""

from __future__ import annotations

import json
import time
from typing import Iterator, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.pipeline import Pipeline
from core.generate import Generator, answer_stats
from api.deps import get_pipeline, get_generator
from api.logging_config import get_logger

log = get_logger("api.chat")
router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="The player's question.")
    history: Optional[str] = Field(default=None, max_length=8000)
    k: int = Field(default=8, ge=1, le=20)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _event_stream(pipeline: Pipeline, generator: Generator,
                  question: str, history: Optional[str], k: int) -> Iterator[str]:
    # This endpoint is sync `def`, so Starlette iterates it in the threadpool —
    # each stream already runs off the event loop (no manual offload needed here).
    t0 = time.perf_counter()
    try:
        result = pipeline.run(question, history=history, k=k)
        yield _sse("meta", {
            "tone": result.decision.tone,
            "used_retrieval": result.retrieved,
            "entity_fallback": result.entity_fallback,
        })
        trace: dict = {}
        parts: list[str] = []
        gen_start = time.perf_counter()
        ttft_ms: Optional[int] = None
        for token in generator.stream(result, history=history, trace=trace):
            if ttft_ms is None:
                ttft_ms = int((time.perf_counter() - gen_start) * 1000)
            parts.append(token)
            yield _sse("token", {"text": token})
        answer = "".join(parts)
        sources, images = Generator.assemble_metadata(result, answer_text=answer)
        yield _sse("sources", {"sources": sources})
        yield _sse("images", {"images": images})
        stats = answer_stats(result, trace.get("model"), {
            "ttft_ms": ttft_ms,
            "gen_ms": int((time.perf_counter() - gen_start) * 1000),
            "total_ms": int((time.perf_counter() - t0) * 1000),
        })
        log.info("chat.done", extra={"fields": {**stats, "question": question[:200]}})
        yield _sse("done", {"stats": stats})
    except Exception:
        log.exception("error while streaming answer")
        yield _sse("error", {"message": "Generation failed. Please try again."})


@router.post("/chat")
def chat(
    req: ChatRequest,
    pipeline: Pipeline = Depends(get_pipeline),
    generator: Generator = Depends(get_generator),
) -> StreamingResponse:
    log.info("chat: q=%r k=%d", req.question[:80], req.k)
    stream = _event_stream(pipeline, generator, req.question, req.history, req.k)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )