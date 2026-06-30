"""
backend/api/db/repository.py

Data-access layer. Endpoints call these functions; they never write raw queries.
Keeps persistence logic in one testable place.
"""

from __future__ import annotations

import uuid
from typing import Optional, Sequence

from sqlalchemy import select, delete, update, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from api.db.models import Conversation, Message


# ── Conversations ─────────────────────────────────────────────────────────────

async def create_conversation(
    db: AsyncSession, user_id: Optional[uuid.UUID],
    title: str = "New conversation", tone: str = "scholar",
) -> Conversation:
    conv = Conversation(user_id=user_id, title=title, tone=tone)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def list_conversations(
    db: AsyncSession, user_id: Optional[uuid.UUID]
) -> Sequence[Conversation]:
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    return (await db.execute(stmt)).scalars().all()


async def get_conversation(
    db: AsyncSession, conv_id: uuid.UUID, user_id: Optional[uuid.UUID],
    with_messages: bool = False,
) -> Optional[Conversation]:
    stmt = select(Conversation).where(
        Conversation.id == conv_id, Conversation.user_id == user_id
    )
    if with_messages:
        stmt = stmt.options(selectinload(Conversation.messages))
    return (await db.execute(stmt)).scalar_one_or_none()


async def rename_conversation(
    db: AsyncSession, conv_id: uuid.UUID, user_id: Optional[uuid.UUID],
    title: Optional[str] = None, tone: Optional[str] = None,
) -> Optional[Conversation]:
    conv = await get_conversation(db, conv_id, user_id)
    if conv is None:
        return None
    if title is not None:
        conv.title = title
    if tone is not None:
        conv.tone = tone
    await db.commit()
    await db.refresh(conv)
    return conv


async def delete_conversation(
    db: AsyncSession, conv_id: uuid.UUID, user_id: Optional[uuid.UUID]
) -> bool:
    conv = await get_conversation(db, conv_id, user_id)
    if conv is None:
        return False
    await db.delete(conv)
    await db.commit()
    return True


# ── Messages ──────────────────────────────────────────────────────────────────

async def add_message(
    db: AsyncSession, conv_id: uuid.UUID, role: str, content: str,
    sources: Optional[list] = None, images: Optional[list] = None,
    tone: Optional[str] = None,
) -> Message:
    msg = Message(
        conversation_id=conv_id, role=role, content=content,
        sources=sources, images=images, tone=tone,
    )
    db.add(msg)
    # Touch the conversation's updated_at so recent convos sort first.
    await db.execute(
        update(Conversation).where(Conversation.id == conv_id)
        .values(updated_at=func.now())
    )
    await db.commit()
    await db.refresh(msg)
    return msg


async def get_recent_messages(
    db: AsyncSession, conv_id: uuid.UUID, limit: int
) -> list[Message]:
    """Most recent `limit` messages, returned in chronological order."""
    stmt = (
        select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return list(reversed(rows))


def format_history(messages: list[Message]) -> str:
    """Render recent messages into the plain-text history the router/generator expect."""
    lines = []
    for m in messages:
        who = "User" if m.role == "user" else "Assistant"
        lines.append(f"{who}: {m.content}")
    return "\n".join(lines)