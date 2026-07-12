"""
backend/api/config.py

Centralized, typed, validated configuration via Pydantic Settings.
One Settings object, populated from environment / .env, used across the API.
Catches missing/invalid config at startup with a clear error rather than a
cryptic failure mid-request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The .env lives at the project root (elden-ring-rag/.env), but the app may be
# launched from backend/ or elsewhere. Resolve it by an absolute path from this
# file's location so config loads regardless of the working directory.
#   this file: elden-ring-rag/backend/api/config.py  →  parents[2] = elden-ring-rag
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore"
    )

    # ── Environment ──────────────────────────────────────────────────────────
    environment: Literal["dev", "prod"] = Field(default="dev", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── API server ───────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=3000, alias="API_PORT")

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Comma-separated list in env, e.g. CORS_ORIGINS="https://x.vercel.app,http://localhost:3000"
    cors_origins: str = Field(default="", alias="CORS_ORIGINS")

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field(default="elden_ring", alias="QDRANT_COLLECTION_NAME")

    # ── LLM providers (presence gates each fallback tier) ────────────────────
    nvidia_api_key: str | None = Field(default=None, alias="NVIDIA_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")

    # ── Database (Supabase Postgres) ─────────────────────────────────────────
    # Use the async driver form, e.g.
    #   postgresql+asyncpg://postgres:[PASSWORD]@db.<ref>.supabase.co:5432/postgres
    database_url: str | None = Field(default=None, alias="DATABASE_URL")

    # ── Conversation history ─────────────────────────────────────────────────
    history_turns: int = Field(default=6, alias="HISTORY_TURNS")  # recent msgs passed as context

    # ── Derived helpers ──────────────────────────────────────────────────────
    @property
    def is_dev(self) -> bool:
        return self.environment == "dev"

    @property
    def cors_allow_origins(self) -> list[str]:
        """
        Dev: permissive (localhost on any port + explicit origins).
        Prod: strict — only the configured origins.
        """
        configured = [o.strip() for o in self.cors_origins.split(",") if o.strip()]
        if self.is_dev:
            # Allow common local dev origins plus anything explicitly configured.
            local = [
                "http://localhost:3000", "http://127.0.0.1:3000",
                "http://localhost:3001", "http://127.0.0.1:3001",
                "http://localhost:5173", "http://127.0.0.1:5173",
            ]
            return list(dict.fromkeys(local + configured))
        return configured


@lru_cache
def get_settings() -> Settings:
    """Cached singleton settings instance."""
    return Settings()