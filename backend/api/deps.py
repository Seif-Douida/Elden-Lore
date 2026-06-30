"""
backend/api/deps.py

Shared, expensive singletons (Pipeline, Generator) built once during the app
lifespan and stored on app.state. Routes access them via FastAPI dependencies,
so handlers stay testable (a test can override these) and nothing is built at
import time.
"""

from __future__ import annotations

from fastapi import Request

from core.pipeline import Pipeline
from core.generate import Generator


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def get_generator(request: Request) -> Generator:
    return request.app.state.generator