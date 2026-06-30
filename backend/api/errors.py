"""
backend/api/errors.py

Consistent error responses and exception handlers. Registered on the app in
main.py so every failure returns a predictable JSON shape.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from api.logging_config import get_logger

log = get_logger("api.errors")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        log.warning("validation error on %s: %s", request.url.path, exc.errors())
        return _error(422, "validation_error", "Invalid request payload.")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Last-resort handler — log full detail, return a safe generic message.
        log.exception("unhandled error on %s", request.url.path)
        return _error(500, "internal_error", "An unexpected error occurred.")