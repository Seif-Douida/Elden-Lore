"""
backend/core/llm.py

The LLM layer: one interface, three providers, transparent fallback.

    NIM  ──fail──▶  Gemini 2.5 Flash  ──fail──▶  Ollama (local)
  primary             cloud fallback               last resort

All three are LangChain BaseChatModels, so .invoke(), .stream(), and
.with_structured_output() behave identically regardless of which tier answers.
Downstream code (router, chat chain, summariser) calls one object and never
needs to know who responded.

Resilience is graceful: a tier is only added to the chain if its credentials
are present. With just NVIDIA_API_KEY set, you get NIM alone. Add GOOGLE_API_KEY
and/or run Ollama, and they slot into the fallback chain automatically — no code
change. Fallback triggers ONLY on retryable errors (rate limit, server error,
timeout, connection) — never on a 400/auth error that would fail everywhere.

Two separate NIM models are used:
  NIM_MODEL        — answer generation (chat). Can be any quality model.
  NIM_ROUTER_MODEL — routing only. MUST support structured output (JSON schema).
                     Defaults to meta/llama-3.1-8b-instruct: fast, reliable, and
                     handles with_structured_output(). Larger NIM free-tier models
                     (e.g. llama-3.3-70b) time out at 60s. Do NOT set this to a
                     model that lacks structured output support (e.g.
                     nemotron-3-ultra-550b-a55b) — those calls hang indefinitely.

Env (.env):
    NVIDIA_API_KEY     required for the NIM primary
    GOOGLE_API_KEY     enables the Gemini fallback (optional)
    OLLAMA_HOST        enables the Ollama last resort (optional, default localhost)
    NIM_MODEL          default: nvidia/nemotron-3-ultra-550b-a55b  (chat)
    NIM_ROUTER_MODEL   default: meta/llama-3.1-8b-instruct            (routing)
    GEMINI_MODEL       default: gemini-2.5-flash
    OLLAMA_MODEL       default: llama3.2:3b
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or None
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or None
OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")

NIM_MODEL        = os.getenv("NIM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
NIM_ROUTER_MODEL = os.getenv("NIM_ROUTER_MODEL", "meta/llama-3.1-8b-instruct")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Whether to attempt the local Ollama tier at all. Off unless explicitly enabled,
# so a machine without Ollama doesn't pay a connection timeout on every fallback.
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() in ("1", "true", "yes")


# ── Retryable exceptions ──────────────────────────────────────────────────────
# Only these trigger a fallback to the next tier. A 400/auth/validation error is
# deterministic — it would fail on every provider — so we let it raise instead of
# wasting time hopping tiers.

def _retryable_exceptions() -> tuple[type[BaseException], ...]:
    excs: list[type[BaseException]] = [
        ConnectionError, TimeoutError,
    ]
    # requests library (used by the NIM/langchain-nvidia-ai-endpoints client)
    try:
        import requests.exceptions as req_exc
        excs += [req_exc.Timeout, req_exc.ReadTimeout, req_exc.ConnectionError,
                 req_exc.ConnectTimeout]
    except Exception:
        pass
    # httpx errors (used by most providers under the hood)
    try:
        import httpx
        excs += [httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError]
    except Exception:
        pass
    # OpenAI-style errors surfaced by the NIM integration
    try:
        from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
        excs += [RateLimitError, APITimeoutError, APIConnectionError, InternalServerError]
    except Exception:
        pass
    # Google API errors
    try:
        from google.api_core.exceptions import (
            ResourceExhausted, ServiceUnavailable, DeadlineExceeded, InternalServerError as GISE,
        )
        excs += [ResourceExhausted, ServiceUnavailable, DeadlineExceeded, GISE]
    except Exception:
        pass
    return tuple(excs)


# ── Tier builders (each returns a model or None if unconfigured) ───────────────

def _build_nim(temperature: float, model: str | None = None):
    if not NVIDIA_API_KEY:
        return None
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
    return ChatNVIDIA(
        model=model or NIM_MODEL,
        api_key=NVIDIA_API_KEY,
        temperature=temperature,
        max_tokens=1024,
    )


def _build_gemini(temperature: float):
    if not GOOGLE_API_KEY:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=temperature,
    )


def _build_ollama(temperature: float):
    if not USE_OLLAMA:
        return None
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        temperature=temperature,
    )


def _assemble_chain(temperature: float, nim_model: str | None = None):
    """
    Build the NIM→Gemini→Ollama chain from whichever tiers are configured.
    The first available tier is the primary; the rest become ordered fallbacks.
    Pass nim_model to override the NIM model (e.g. NIM_ROUTER_MODEL for routing).
    """
    tiers = [t for t in (
        _build_nim(temperature, model=nim_model),
        _build_gemini(temperature),
        _build_ollama(temperature),
    ) if t is not None]

    if not tiers:
        raise RuntimeError(
            "No LLM provider configured. Set NVIDIA_API_KEY (and optionally "
            "GOOGLE_API_KEY / USE_OLLAMA) in your .env."
        )

    primary, fallbacks = tiers[0], tiers[1:]
    if not fallbacks:
        return primary
    return primary.with_fallbacks(
        fallbacks,
        exceptions_to_handle=_retryable_exceptions(),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def get_chat_llm(temperature: float = 0.7):
    """General chat / streaming model (used by the chat endpoint, summariser)."""
    return _assemble_chain(temperature)


def get_router_llm(temperature: float = 0.0):
    """
    Deterministic model for the agent router's structured output.
    Uses NIM_ROUTER_MODEL (default: meta/llama-3.1-8b-instruct), which is fast
    and reliably handles with_structured_output(). temperature=0 for stable JSON.
    """
    return _assemble_chain(temperature, nim_model=NIM_ROUTER_MODEL)


def configured_tiers() -> list[str]:
    """Which tiers are active, in fallback order — handy for a /health check."""
    out = []
    if NVIDIA_API_KEY:
        out.append(f"nim:{NIM_MODEL}")
        out.append(f"nim-router:{NIM_ROUTER_MODEL}")
    if GOOGLE_API_KEY:
        out.append(f"gemini:{GEMINI_MODEL}")
    if USE_OLLAMA:
        out.append(f"ollama:{OLLAMA_MODEL}")
    return out


# ── CLI smoke test ────────────────────────────────────────────────────────────

def _smoke_test(prompt: str) -> None:
    from rich.console import Console
    console = Console()

    tiers = configured_tiers()
    console.print(f"[bold]Configured tiers:[/bold] {tiers or '[red]none[/red]'}")
    if not tiers:
        console.print("[red]Set NVIDIA_API_KEY in .env to test.[/red]")
        return

    console.print(f"[dim]Prompt: {prompt}[/dim]\n")
    llm = get_chat_llm()

    console.print("[bold cyan]Streaming response:[/bold cyan]")
    got = False
    for chunk in llm.stream(prompt):
        text = getattr(chunk, "content", "")
        if text:
            got = True
            console.print(text, end="")
    console.print()
    if not got:
        console.print("[yellow]No tokens streamed — check the model/key.[/yellow]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Smoke-test the LLM layer")
    parser.add_argument("prompt", nargs="*",
                        default=["Say hello as a wise Elden Ring sage in one sentence."])
    args = parser.parse_args()
    _smoke_test(" ".join(args.prompt) if args.prompt else
                "Say hello as a wise Elden Ring sage in one sentence.")