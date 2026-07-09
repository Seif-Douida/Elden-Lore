"""
backend/api/logging_config.py

Structured JSON logging. Call configure_logging() once at app startup.

On Cloud Run, each stdout line that is a JSON object with a `severity` field is
parsed into a structured `jsonPayload` log entry, so extra fields (model, timings,
intent, …) become filterable in Logs Explorer. Locally it's still one JSON line
per record — greppable, and `jq`-able.

Attach structured fields with the standard logging `extra=`:
    log.info("chat.request", extra={"fields": {"total_ms": 812, "model": "gemini"}})
Everything under `fields` is merged into the top-level JSON object.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sys

# Standard LogRecord attributes we don't want to duplicate into the JSON body.
_RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime", "fields"}

# Python level name → Cloud Logging severity.
_SEVERITY = {
    "DEBUG": "DEBUG", "INFO": "INFO", "WARNING": "WARNING",
    "ERROR": "ERROR", "CRITICAL": "CRITICAL",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "severity": _SEVERITY.get(record.levelname, "DEFAULT"),
            "time": _dt.datetime.fromtimestamp(
                record.created, _dt.timezone.utc
            ).isoformat(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge a structured `fields` dict if present.
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        # Also pick up any ad-hoc extras passed directly (extra={"foo": ...}).
        for k, v in record.__dict__.items():
            if k not in _RESERVED and k not in payload:
                payload[k] = v
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers on uvicorn --reload re-imports.
    root.handlers = [handler]

    # Tame noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
