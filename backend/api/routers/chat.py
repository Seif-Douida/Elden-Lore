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
from typing import Iterator, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.pipeline import Pipeline
from core.generate import Generator
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
    try:
        result = pipeline.run(question, history=history, k=k)
        yield _sse("meta", {
            "tone": result.decision.tone,
            "used_retrieval": result.retrieved,
            "entity_fallback": result.entity_fallback,
        })
        for token in generator.stream(result, history=history):
            yield _sse("token", {"text": token})
        sources, images = Generator.assemble_metadata(result)
        yield _sse("sources", {"sources": sources})
        yield _sse("images", {"images": images})
        yield _sse("done", {})
    except Exception as e:  # noqa: BLE001
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