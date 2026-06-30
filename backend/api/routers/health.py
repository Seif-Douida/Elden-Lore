"""
backend/api/routers/health.py

Liveness/readiness endpoint. Reports configured LLM tiers and whether the
heavy singletons are ready.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from core.llm import configured_tiers

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    ready = (
        getattr(request.app.state, "pipeline", None) is not None
        and getattr(request.app.state, "generator", None) is not None
    )
    return {
        "status": "ok",
        "llm_tiers": configured_tiers(),
        "ready": ready,
    }