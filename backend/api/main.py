"""
backend/api/main.py

The FastAPI app factory. Builds the expensive singletons (Pipeline, Generator)
once in the lifespan, configures CORS from settings, registers error handlers,
and mounts the routers.

Run:
    cd backend
    uv run uvicorn api.main:app --reload --port 3000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.logging_config import configure_logging, get_logger
from api.errors import register_exception_handlers
from api.routers import chat as chat_router
from api.routers import health as health_router

log = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: build heavy singletons once ─────────────────────────────────
    settings = get_settings()
    configure_logging("INFO")
    log.info("starting API (env=%s) — building pipeline & generator…", settings.environment)

    # Imported here so app import stays cheap and testable.
    from core.pipeline import Pipeline
    from core.generate import Generator

    app.state.pipeline = Pipeline()
    app.state.generator = Generator()

    # Build the DB engine if a database is configured (persistence is optional
    # in pure-retrieval/dev runs).
    from api.db.session import build_engine, dispose_engine
    if settings.database_url:
        build_engine()
        log.info("database engine ready")
    else:
        log.warning("DATABASE_URL not set — conversation persistence disabled")

    log.info("ready — tiers=%s", _safe_tiers())

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────────
    log.info("shutting down")
    from api.db.session import dispose_engine
    await dispose_engine()
    app.state.pipeline = None
    app.state.generator = None


def _safe_tiers():
    try:
        from core.llm import configured_tiers
        return configured_tiers()
    except Exception:
        return []


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Elden Ring RAG API", version="1.0.0", lifespan=lifespan)

    # CORS — permissive in dev, strict allowlist in prod (from settings).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(health_router.router, tags=["health"])
    app.include_router(chat_router.router, tags=["chat"])
    from api.routers import conversations as conversations_router
    app.include_router(conversations_router.router)
    return app


app = create_app()