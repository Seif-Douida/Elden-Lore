# syntax=docker/dockerfile:1
# Backend image for GCP Cloud Run — CPU-only (no GPU at serving time).
# Context = repo root (needs pyproject.toml + uv.lock, which live here, not in backend/).
FROM python:3.12-slim

# libgomp1 = OpenMP runtime that torch's CPU wheel loads at import time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# uv (static binaries from the official image).
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/huggingface \
    EMBED_DEVICE=cpu \
    ENVIRONMENT=prod \
    PORT=8080

WORKDIR /app

# 1) Install locked deps EXCEPT torch — the lock pins torch 2.5.1+cu121 (a ~2.5GB
#    CUDA build, useless on CPU Cloud Run). Then add the matching CPU wheel; the
#    +cpu tag only exists on the pytorch cpu index, so it can't resolve to CUDA.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-package torch \
 && uv pip install --python /app/.venv/bin/python "torch==2.5.1+cpu" \
      --extra-index-url https://download.pytorch.org/whl/cpu

# 2) App code — backend package only (api/ + core/, incl. core/data/gazetteer.json).
#    The serving path never imports data_pipeline/.
COPY backend/ ./backend/

# 3) Pre-bake the bge embedder so the first request doesn't pay a ~400MB download.
#    Call the venv python directly (NOT `uv run`) so uv doesn't re-sync and drag
#    the CUDA torch back in.
RUN /app/.venv/bin/python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

WORKDIR /app/backend
# Cloud Run injects $PORT (default 8080). Bind to it; run uvicorn from the venv.
CMD ["sh", "-c", "/app/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
