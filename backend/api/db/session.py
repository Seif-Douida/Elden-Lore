"""
backend/api/db/session.py

Async SQLAlchemy engine + session factory, pointed at Supabase Postgres via the
asyncpg driver. A FastAPI dependency yields a session per request.
"""

from __future__ import annotations

from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession, async_sessionmaker, create_async_engine,
)

from api.config import get_settings

_settings = get_settings()

# Engine is created lazily so the app can import without a DB (e.g. for tests
# that don't touch persistence). build_engine() is called from the lifespan.
_engine = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def build_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        return
    if not _settings.database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add your Supabase async connection string "
            "(postgresql+asyncpg://...) to .env."
        )
    _engine = create_async_engine(
        _settings.database_url,
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,   # validate connections (Supabase closes idle ones)
        echo=False,
    )
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: yields a session, ensures cleanup."""
    if _sessionmaker is None:
        build_engine()
    assert _sessionmaker is not None
    async with _sessionmaker() as session:
        yield session