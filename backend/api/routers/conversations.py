"""
backend/api/routers/conversations.py

Conversation CRUD + the conversation-aware streaming chat.

The chat flow:
  1. load recent history from the DB → context for router/generator
  2. save the user's message immediately (survives a failed generation)
  3. stream the answer (SSE), accumulating the text
  4. on completion, persist the assistant message with sources + images

Persistence timing rationale: user msg saved up front so the turn is recorded
even on failure; assistant msg saved at the end when we have the full text +
metadata (no per-token DB writes).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.pipeline import Pipeline
from core.generate import Generator, answer_stats
from api.deps import get_pipeline, get_generator
from api.db.session import get_session
from api.db import repository as repo
from api.auth import get_current_user
from api.config import get_settings
from api.logging_config import get_logger

log = get_logger("api.conversations")
router = APIRouter(prefix="/conversations", tags=["conversations"])
_settings = get_settings()


# ── Schemas ───────────────────────────────────────────────────────────────────

class CreateConversation(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    tone: str = Field(default="scholar")


class UpdateConversation(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    tone: Optional[str] = None


class ChatMessage(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=8, ge=1, le=20)
    tone: Optional[str] = None  # optional per-message override


def _conv_out(conv) -> dict:
    return {
        "id": str(conv.id), "title": conv.title, "tone": conv.tone,
        "created_at": conv.created_at.isoformat(),
        "updated_at": conv.updated_at.isoformat(),
    }


def _msg_out(m) -> dict:
    return {
        "id": str(m.id), "role": m.role, "content": m.content,
        "sources": m.sources, "images": m.images, "tone": m.tone,
        "created_at": m.created_at.isoformat(),
    }


# ── CRUD ──────────────────────────────────────────────────────────────────────

@router.post("")
async def create(
    body: CreateConversation,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict:
    conv = await repo.create_conversation(
        db, user_id, title=body.title or "New conversation", tone=body.tone
    )
    return _conv_out(conv)


@router.get("")
async def list_all(
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> list[dict]:
    convs = await repo.list_conversations(db, user_id)
    return [_conv_out(c) for c in convs]


@router.get("/{conv_id}")
async def get_one(
    conv_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict:
    conv = await repo.get_conversation(db, conv_id, user_id, with_messages=True)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    return {**_conv_out(conv), "messages": [_msg_out(m) for m in conv.messages]}


@router.patch("/{conv_id}")
async def update(
    conv_id: uuid.UUID, body: UpdateConversation,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict:
    conv = await repo.rename_conversation(db, conv_id, user_id, body.title, body.tone)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    return _conv_out(conv)


@router.delete("/{conv_id}")
async def remove(
    conv_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict:
    ok = await repo.delete_conversation(db, conv_id, user_id)
    if not ok:
        raise HTTPException(404, "Conversation not found")
    return {"deleted": True}


# ── Conversation-aware streaming chat ─────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/{conv_id}/chat")
async def conversation_chat(
    conv_id: uuid.UUID, body: ChatMessage,
    user_id: uuid.UUID = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    pipeline: Pipeline = Depends(get_pipeline),
    generator: Generator = Depends(get_generator),
) -> StreamingResponse:
    conv = await repo.get_conversation(db, conv_id, user_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")

    # 1) history from recent messages
    recent = await repo.get_recent_messages(db, conv_id, _settings.history_turns)
    history = repo.format_history(recent) if recent else None

    # 2) save the user's message immediately
    await repo.add_message(db, conv_id, role="user", content=body.question)

    # Auto-title a fresh conversation from its first question.
    if conv.title == "New conversation":
        await repo.rename_conversation(
            db, conv_id, user_id, title=body.question[:80]
        )

    async def event_gen():
        t0 = time.perf_counter()
        # Offload the blocking router+retrieval off the event loop so concurrent
        # users don't serialize behind each other.
        result = await run_in_threadpool(
            pipeline.run, body.question, history=history, k=body.k
        )
        # tone: explicit override → else router's pick
        if body.tone:
            result.decision.tone = body.tone

        yield _sse("meta", {
            "tone": result.decision.tone,
            "used_retrieval": result.retrieved,
            "entity_fallback": result.entity_fallback,
        })

        trace: dict = {}
        parts: list[str] = []
        gen_start = time.perf_counter()
        ttft_ms: Optional[int] = None
        try:
            # Iterate the sync token generator IN THE THREADPOOL so the loop stays
            # free between tokens (interleaves other users' streams).
            async for token in iterate_in_threadpool(
                generator.stream(result, history=history, trace=trace)
            ):
                if ttft_ms is None:
                    ttft_ms = int((time.perf_counter() - gen_start) * 1000)
                parts.append(token)
                yield _sse("token", {"text": token})

            answer = "".join(parts)
            sources, images = Generator.assemble_metadata(result, answer_text=answer)
            yield _sse("sources", {"sources": sources})
            yield _sse("images", {"images": images})
            # 4) persist the assistant message with full text + metadata
            await repo.add_message(
                db, conv_id, role="assistant", content=answer,
                sources=sources, images=images, tone=result.decision.tone,
            )
            stats = answer_stats(result, trace.get("model"), {
                "ttft_ms": ttft_ms,
                "gen_ms": int((time.perf_counter() - gen_start) * 1000),
                "total_ms": int((time.perf_counter() - t0) * 1000),
            })
            log.info("chat.done", extra={"fields": {
                **stats, "conv_id": str(conv_id), "user_id": str(user_id),
                "question": body.question[:200],
            }})
            yield _sse("done", {"stats": stats})
        except Exception:
            log.exception("generation failed mid-stream",
                          extra={"fields": {"conv_id": str(conv_id)}})
            # still persist whatever was produced, marked partial
            if parts:
                await repo.add_message(
                    db, conv_id, role="assistant", content="".join(parts),
                    tone=result.decision.tone,
                )
            yield _sse("error", {"message": "Generation failed. Please try again."})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )