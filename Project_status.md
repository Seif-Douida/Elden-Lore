# Elden Path — Project State Summary

A portfolio **RAG chatbot for Elden Ring** (lore, items, quests, bosses, locations).
User asks questions; the bot answers from retrieved Fextralife wiki context, shows
images, and supports two voices: **Scholar** (factual, strictly grounded) and
**Cryptic** (atmospheric VaatiVidya-style lore). Full stack is built and running
locally end to end. This document is the handoff for continuing in a new chat.

---

## USER CONTEXT
- Experienced engineer on **Windows**, uses **UV** (Python) + **npm** (Node).
- GPU: **GTX 1650 (4GB VRAM)**, CUDA-enabled for embedding.
- Works carefully, tests each stage, shares output files back. Prefers discussing
  design/tradeoffs before building. Likes the assistant to use the
  `ask_user_input` tool for design decisions.
- Backend runs on **port 3000**; frontend dev server on **port 3001**.
- Node updated to ≥20.9 (was 20.2, which broke Next.js).

## WORKING STYLE
- Deliverable code is produced in `/mnt/user-data/outputs/`, then the user copies
  files into the project tree. The container here has NO network/Qdrant/GPU; the
  assistant syntax-checks and validates logic against uploaded data; the user runs
  actual execution and shares results.

---

## TECH STACK (all decided + implemented)
- **Frontend**: Next.js (App Router) + TypeScript + Tailwind, deployed target Vercel.
  Fonts: **Marcellus** (display) + **Spectral** (body). React Query for server
  state. Custom SSE streaming hook.
- **Backend**: FastAPI, **LangChain** for LLM/retrieval glue. Deployed target GCP Cloud Run.
- **Vector DB**: **Qdrant** in local Docker (qdrant/qdrant:v1.16.1). CRITICAL: uses a
  Docker **NAMED VOLUME** `qdrant_storage`, NOT a Windows bind mount (see corruption
  saga below).
- **LLM layer**: three-tier fallback **NIM (Llama 3.1 8B) → Gemini 2.5 Flash → Ollama**.
  Credential-gated (present key = active tier). Currently only **Gemini active**
  (NVIDIA key was pending; Gemini works great).
- **Embedder**: **BAAI/bge-base-en-v1.5** (768-dim), local on GPU. Query instruction
  prefix: "Represent this sentence for searching relevant passages:".
- **Auth + DB**: **Supabase** (Postgres). SQLAlchemy 2.0 async + asyncpg, Alembic
  migrations. Auth deferred (stub dev user, JWT-ready).

---

## PROJECT STRUCTURE
```
elden-ring-rag/
├── docker-compose.yml          # Qdrant, NAMED VOLUME, stop_grace_period 120s
├── pyproject.toml              # torch cu121; +fastapi,uvicorn,langchain-*,rapidfuzz,
│                               #   jellyfish,sqlalchemy[asyncio],asyncpg,alembic,
│                               #   pydantic-settings,langchain-google-genai,
│                               #   langchain-nvidia-ai-endpoints,langchain-ollama
├── .env                        # QDRANT_*, GOOGLE_API_KEY, DATABASE_URL (Supabase
│                               #   SESSION POOLER url w/ +asyncpg), ENVIRONMENT=dev
├── alembic/
│   ├── env.py                  # async-wired to api.db.models  (from alembic_env.py)
│   └── versions/               # initial migration created (conversations+messages)
├── data_pipeline/
│   ├── common.py, discover.py, scrape.py, chunker.py
│   ├── embed.py                # Phase 4a: embed → data/embeddings/{vectors.npy,ids.npy,
│   │                           #   payloads.jsonl,meta.json}
│   ├── upload.py               # Phase 4b: load embeddings → Qdrant (hardened)
│   └── data/                   # discovered_urls.jsonl, pages.jsonl, chunks.jsonl, embeddings/
└── backend/
    ├── core/
    │   ├── __init__.py
    │   ├── retriever.py        # vector + filters + structural boost  (imports core.x)
    │   ├── llm.py              # 3-tier fallback
    │   ├── entity_resolver.py  # rapidfuzz + phonetic resolution
    │   ├── build_gazetteer.py  # builds data/gazetteer.json (4,872 entities)
    │   ├── router.py           # 2-call agent router (index-based Call 2)
    │   ├── pipeline.py         # router → retriever orchestrator
    │   ├── prompts.py          # tone templates + context/image assembly
    │   ├── generate.py         # streaming generation + Assistant
    │   └── data/gazetteer.json
    ├── api/
    │   ├── __init__.py
    │   ├── main.py             # app factory: lifespan, CORS, errors, routers
    │   ├── config.py           # Pydantic Settings (loads .env by ABSOLUTE path)
    │   ├── deps.py             # get_pipeline/get_generator from app.state
    │   ├── errors.py           # exception handlers
    │   ├── logging_config.py
    │   ├── auth.py             # get_current_user STUB (dev user; JWT later)
    │   ├── db/
    │   │   ├── __init__.py
    │   │   ├── session.py      # async engine + session factory
    │   │   ├── models.py       # Conversation, Message (SQLAlchemy)
    │   │   └── repository.py   # CRUD + history assembly
    │   └── routers/
    │       ├── __init__.py
    │       ├── chat.py         # standalone /chat SSE (from router_chat.py)
    │       ├── conversations.py# CRUD + conversation-aware /chat (from router_conversations.py)
    │       └── health.py       # /health (from router_health.py)
    └── ...
frontend/
├── .env.local                  # NEXT_PUBLIC_API_URL=http://localhost:3000
├── public/sigil.jpg            # the uploaded Elden Ring sigil image
├── app/
│   ├── layout.tsx              # Marcellus + Spectral via next/font
│   ├── page.tsx                # sidebar + chat orchestration + empty state
│   ├── providers.tsx           # React Query
│   └── globals.css             # theme tokens + markdown/answer styles
├── components/
│   ├── ChatView.tsx            # streaming orchestrator
│   ├── Sidebar.tsx, ChatInput.tsx, VoiceToggle.tsx, SigilBackground.tsx
│   └── Message.tsx             # markdown + sources + image gallery
└── lib/
    ├── types.ts                # API contract types
    ├── api.ts                  # CRUD client
    └── useChatStream.ts        # SSE streaming hook w/ typewriter reveal
```
NOTE: output files `router_chat.py`/`router_conversations.py`/`router_health.py`
map to `backend/api/routers/chat.py`/`conversations.py`/`health.py`.
`alembic_env.py` → `alembic/env.py`. `EldenPathMockup.jsx`/`ArchiveMockup.jsx`
were design mockups (superseded by the real frontend components).

---

## DATA PIPELINE — COMPLETE (23,351 vectors live in Qdrant)
- **discover.py**: breadcrumb-recursive DFS, 4 entry points, 3,789 pages discovered.
  Excludes `table.wiki_table` links during discovery — this caused the KNOWN GAP
  (missing katanas, see below).
- **scrape.py** → pages.jsonl (3,789 records). Extractor cleans infobox/dialogue/body,
  has `doc_type` (page|walkthrough).
- **chunker.py** → chunks.jsonl (**23,351 chunks**). Section-aware split on `##`,
  sentence-pack to 350 tokens (bge), context-prefix, spaCy PhraseMatcher entity
  tagging (cap 30/chunk).
- **chunks.jsonl schema**: chunk_id, text (embedded), raw_text (display), url, title,
  category (lore/item/boss/quest/location), doc_type, breadcrumb[], section_heading,
  chunk_type (body/dialogue/item_desc), entities[], image_url, source_type.
- **embed/upload split** [KEY DECISION]: decoupled slow GPU embedding (durable .npy
  artifact) from fast retryable upload. embed.py does sanity checks (NaN/inf/zero-norm/
  collisions/norm≈1.0). upload.py is hardened (single optimization thread, wait=True,
  wait-for-green poll, snapshot after green).

## QDRANT CORRUPTION SAGA — RESOLVED
Repeated `OutputTooSmall { expected: 4, actual: 0 }` panic on payload reads. Chased
two false theories (interrupted optimizer). TRUE ROOT CAUSE: startup log
"Filesystem check failed... Unrecognized filesystem" — Qdrant's gridstore uses
**memory-mapped I/O which is unreliable across the Docker-Desktop/WSL2 bind mount
from Windows**. FIX: switched docker-compose volume from bind mount
`./qdrant_storage:/...` to **Docker NAMED VOLUME** `qdrant_storage:/...` (lives in
WSL2 ext4, mmap-safe). The embed/upload split made recovery a 30-sec re-upload.

---

## RETRIEVER — COMPLETE, LOCKED (backend/core/retriever.py)
Thin retriever on raw qdrant-client reading FLAT payload. Design:
1. **dense vector search** (bge) for recall
2. **entity + chunk_type/category/doc_type filters** for precision — THE real engine
3. **structural boost** for within-set ordering (NOT a neural reranker)

**Neural cross-encoder reranker was TESTED AND DROPPED** — it over-weighted vocabulary
overlap (e.g. boosted Sorceress Sellen for a Ranni question), hurting entity/dialogue
queries. Replaced with **additive structural boost** (small weights so vector score
dominates): SECTION_BOOST 0.05, BREADCRUMB_BOOST 0.03, CHUNKTYPE_BOOST 0.05,
DIVERSITY_PENALTY 0.04. INTENT_PROFILES for drops/strategy/location/dialogue/lore/quest
(section keywords + crumb keywords + chunk_types). fetch_k=40 → re-sort → return k=8.
Stress-tested: wrong-intent only mildly reshuffles (router needn't be perfect);
entity-filtering alone beat the neural reranker on every problem query.

## LLM LAYER — COMPLETE (backend/core/llm.py)
3-tier `.with_fallbacks(exceptions_to_handle=...)` — only retryable errors (rate
limit/server/timeout/connection) trigger fallback, not 400/auth. Tier builders return
None if creds absent (graceful). get_chat_llm(temp 0.7), get_router_llm(temp 0.0),
configured_tiers(). Gemini tested working.

## ENTITY RESOLVER — COMPLETE (entity_resolver.py + build_gazetteer.py)
Shortlist approach: rapidfuzz multi-scorer (WRatio, token_sort, partial_ratio,
**phonetic via jellyfish Metaphone** + consonant-skeleton fallback) → top-5 shortlist;
router LLM picks. **Phonetic scorer was THE fix** for "Melania"→Malenia (edit-distance
ranked Melina higher; phonetic groups them). Gazetteer: 4,872 entities (name→frequency)
from chunks.jsonl. SCORE_FLOOR=60, freq tiebreak. Validated: Melania→Malenia(100)|
Melina(100), marika→Queen Marika(100), godrick→Godrick the grafted(100), radan→Radahn.

## ROUTER — COMPLETE (backend/core/router.py)
Two LLM calls + latency guard. Call 1 (.with_structured_output(RouterDecision)):
needs_retrieval, entity_mentions[] (VERBATIM incl typos), intent (Literal), 
chunk_type_bias, category_hint, needs_image, tone. Resolver → shortlists.
**Call 2 is INDEX-BASED** [KEY FIX]: LLM returned off-list "Queen Marika the Eternal"
when asked for names; now shown NUMBERED candidates, returns choice_index, mapped back
to exact string — structurally impossible to paraphrase. Call 2 SKIPPED when no
mentions or single dominant candidate (AUTOACCEPT_SCORE=90, beats runner-up by ≥10).
Output: ResolvedDecision (entity_focus canonical, intent, filters, tone, debug).

## PIPELINE — COMPLETE (backend/core/pipeline.py)
Pure orchestrator. entity_focus = HARD filter with graceful FALLBACK (if entity filter
returns empty, retry WITHOUT entity filter keeping intent/boosts, log note). Boosts =
all four when intent present, diversity-only when not. needs_retrieval=False
short-circuits. Returns PipelineResult(question, decision, chunks, retrieved,
entity_fallback, notes).

## GENERATION — COMPLETE (prompts.py + generate.py)
- **prompts.py**: format_context (labels chunks, cap 6000 chars), format_sources
  (deduped UI cards), **select_images** (entity-driven, contribution-ranked, dynamic
  count; TIGHTENED: non-entity page needs ≥2 chunks or score≥0.74; cap 4; skips
  non-visual pages). Tones: SCHOLAR_SYSTEM (strict grounding), CRYPTIC_RESTRAINED
  (DEFAULT, atmospheric no roleplay) vs CRYPTIC_STRONG (full VaatiVidya, kept for
  optional sub-tone). **All prompts now tell the model the interface auto-displays
  images** — never say "I can't show images" (fixed the Ranni "I'm text-based" bug).
  Grounding: strict Scholar, interpretive Cryptic.
- **generate.py**: pure Generator.stream(PipelineResult) yields tokens; assemble_metadata
  → (sources, images). Assistant full-stack convenience.

## API — COMPLETE, PRODUCTION-STRUCTURED
- Package: api/main.py (app factory, lifespan builds singletons into app.state, CORS
  dev-permissive/prod-strict, error handlers, mounts routers).
- config.py: Pydantic Settings, loads .env by ABSOLUTE path (parents[2]) so it works
  from any CWD. CORS dev origins include localhost 3000/3001/5173.
- SSE event protocol: meta → token×N → sources → images → done (error on failure).
- Conversation-aware chat: loads history, saves user msg immediately, streams, saves
  assistant msg on completion (with sources/images). Auto-titles from first question.
- DB: SQLAlchemy async models (Conversation, Message), repository layer, Alembic
  migrations applied to Supabase. user_id nullable (auth deferred).

## FRONTEND — COMPLETE, RUNNING
Theme "Elden Path": gold-on-near-black, Marcellus + Spectral, sigil watermark,
collapsible sidebar, top-right voice toggle, rounded corners, rune-marked answers,
arrow send button, placeholder "What would you like to know". 
- useChatStream.ts: POST SSE via fetch+ReadableStream (EventSource can't POST), robust
  chunk parsing, **smooth typewriter reveal** (CHARS_PER_FRAME=3, decouples display
  from network bursts; sources/images deferred until text finishes).
- Message.tsx: react-markdown rendering, source cards, image gallery (filters
  empty/broken image URLs).
- React Query for conversations CRUD. Optimistic pending-question display during stream.

---

## ── SUPABASE / DEPLOYMENT NOTES (hard-won) ──
- DATABASE_URL must use **+asyncpg** driver: `postgresql+asyncpg://...`
- Supabase direct host (`db.<ref>.supabase.co:5432`) is **IPv6-only** and failed to
  connect (user's network = IPv4). FIX: use the **Session Pooler** connection string
  (`...pooler.supabase.com:5432`, username `postgres.<ref>`). This is also better for
  Cloud Run later.
- Windows reserved port 8000 (Hyper-V/Docker excluded range) → backend runs on 3000.

---

## CURRENT STATE: FULL STACK WORKING END TO END
User has the complete product running: themed streaming chat, two voices, sources,
images, conversation persistence, multi-turn history (pronoun "what does he drop?"
correctly resolves to Radahn). Recently fixed: empty image boxes, tangential image
over-selection, the "I can't show images" prompt bug, and jumpy streaming (now smooth
typewriter). Last thing tested = the typewriter reveal (awaiting user's feel-check /
possible CHARS_PER_FRAME tuning).

---

## REMAINING TASKS (in user's chosen order: build-to-demo → polish → deploy)

### IMMEDIATE / IN PROGRESS
1. **Confirm typewriter streaming feel** — user testing CHARS_PER_FRAME=3; may tune
   2–5 to taste. Single constant at top of useChatStream.ts.

### THE DATA GAP (user said: LAST polish item)
2. **Index-table discovery gap** — Moonveil, Dragonscale Blade, Serpentbone Blade
   (and likely other weapons/items) are MISSING from the corpus because they're only
   reachable via excluded `table.wiki_table` index tables. Symptom: "where is Moonveil"
   honestly says it can't find the location but answers about its skill (Transient
   Moonlight) from related chunks. FIX: supplementary discovery pass harvesting weapon/
   item index tables → scrape → chunk → embed → upload just new pages (cheap via the
   embed/upload split).

### REMAINING BUILD-TO-DEMO → DEPLOY
3. **Auth wiring** — flip auth.py stub to real Supabase JWT verification; connect
   frontend Supabase session so requests carry tokens. (Schema already has user_id.)
4. **Inline images second pass** (deferred feature) — images interleaved BETWEEN lore
   topics as text streams, via LLM-emitted markers like `[[image: Queen Marika]]`
   replaced while streaming (Option B from discussion). Currently images are a final
   event after text.
5. **Deployment** — backend → Cloud Run (containerize; Qdrant hosting decision: managed
   Qdrant Cloud vs self-host on VM; secrets/env; ENVIRONMENT=prod with strict
   CORS_ORIGINS=<vercel domain>). Frontend → Vercel. Use Supabase session pooler URL.

### KNOWN MINOR / DEFERRED QUALITY ITEMS (logged, non-blocking)
6. **Multi-tagged entity variants** — same character under multiple entity tags
   (Queen Marika 248 vs "Queen Marika the Eternal, Goddess..." 26). Chosen fix =
   Approach A: query-time multi-variant OR, CONSERVATIVE (only group clear same-entity
   like exact-prefix; NEVER merge distinct like Starscourge Radahn vs Promised Consort
   Radahn — different boss fights). No re-ingest. Symptom: character's own page chunk
   ranks below tangential mentions; lore entity's own portrait not guaranteed first image.
7. **Cross-boss boost leak** — Promised Consort Radahn appears in Starscourge Radahn
   results (shared entity tags). Minor.
8. **Gazetteer casing/typo noise** — "Godrick the grafted" (lowercase), "Marika's
   soreseal". Would need normalization pass; matters when filtering Qdrant by exact
   canonical name.
9. **Gemini free-tier rate limit** (10 RPM / 250 daily) — router(1-2) + generation(1)
   per question; hit 429s during heavy testing. Eases with NVIDIA NIM (40 RPM) when
   that key arrives, or space out tests.
10. **Cosmetic**: Qdrant "(unhealthy)" healthcheck false alarm (image lacks curl);
    "Api key used with insecure connection" warning (harmless).

---

## DATABASE SCHEMA (Supabase — built via Alembic)
- conversations(id uuid pk, user_id uuid null idx, title, tone, created_at, updated_at)
- messages(id uuid pk, conversation_id uuid fk cascade idx, role, content, sources jsonb,
  images jsonb, tone, created_at)
- (user_memory designed but NOT built — future cross-conversation personalization)

## KEY COMMANDS
```powershell
# Backend (from backend/)
uv run uvicorn api.main:app --reload --port 3000
# DB migrations (from backend/, after model changes)
uv run alembic revision --autogenerate -m "msg"
uv run alembic upgrade head
# Frontend (from frontend/)
npm run dev -- -p 3001
# Qdrant
docker compose up -d        # NEVER docker rm -f; use docker compose stop
# Re-ingest (cheap, from data_pipeline/)
uv run python data_pipeline/embed.py     # only if chunks changed
uv run python data_pipeline/upload.py --recreate
```

## WINDOWS/POWERSHELL NOTES
- `curl` = Invoke-WebRequest alias (breaks -H/-d). Use `curl.exe` or build JSON with
  `@{...} | ConvertTo-Json` + `Invoke-RestMethod` (handles spaces/quotes safely).
- Core modules now use package imports (`from core.x import ...`) so their standalone
  CLI harnesses run via `python -m core.router` from backend/, not direct file path.