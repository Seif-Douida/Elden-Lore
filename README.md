# Elden Path — an Elden Ring lore & guide chatbot (RAG)

A retrieval-augmented chatbot that answers questions about **Elden Ring** and the **Shadow of the Erdtree** DLC — lore, boss strategies, item locations, questlines, weapon stats, and "how many / which is the best" relational queries — grounded in the community wiki so it doesn't hallucinate.

> **Live demo:** [https://elden-path.vercel.app/]

Ask it *"How do I access the DLC?"*, *"What is the heaviest armor set?"*, *"Give me the stats of the Bloodhound's Fang"*, *"Who is Radahn?"*, or even *"How many weapons are there?"* — and it answers from the wiki, in-character, with the relevant item/boss images shown alongside.

---

## Features

- **Grounded answers, not hallucinations** — every factual claim is grounded in retrieved wiki passages; when the wiki doesn't cover something, it says so instead of inventing.
- **Relational & counting queries** — *"how many talismans"*, *"heaviest armor"*, *"cheapest sorcery"*, *"weapons that scale with Dex"*, *"the 9 legendary armaments"*. These are answered by **aggregating structured metadata**
- **Typo & phonetic tolerance** — *"Melania"* → Malenia, *"moon light greatsword"* → Dark Moon Greatsword, via a fuzzy + phonetic (Metaphone) entity resolver over a wiki gazetteer.
- **Two tones** — a **Scholar** voice (precise, grounded) and an optional **Cryptic** voice
- **Multi-turn follow-ups** — *"How do I reach Rykard?"* → *"What does he drop?"* resolves the referent from conversation history.
- **Auto-displayed images** — the relevant item/boss/location art is selected and shown with each answer.
- **Streaming responses** — token streaming over SSE with a typewriter reveal.
- **Guardrails** — off-topic requests and attempts are deflected in-character.

---

## How it works

```
Question
   │
   ▼
┌─────────────┐   Gemma (structured JSON): intent · entities · query rewrite
│   Router    │   · relational facets · tone. Two calls: extract/route, then an
└─────────────┘   INDEX-BASED entity disambiguation (picks a numbered candidate).
   │
   ▼
┌─────────────┐   Fuzzy + phonetic (Metaphone) match against a wiki gazetteer,
│  Resolver   │   so misspelled/aliased names snap to the canonical entity.
└─────────────┘
   │
   ▼
┌─────────────┐   bge bi-encoder vector search over Qdrant + payload filters
│  Retriever  │   (category / entity / facets) + a small ADDITIVE structural
└─────────────┘   boost (title/section/breadcrumb match). No neural reranker.
   │              For relational/enumeration/superlative queries, a precise
   │              roster is built from Qdrant metadata (counts, lists, rankings).
   ▼
┌─────────────┐   Gemma streams the answer grounded ONLY in retrieved context,
│  Generator  │   in the chosen tone. Sources + images assembled from metadata.
└─────────────┘
```

**Design choices worth calling out**

- **No neural reranker.** A cross-encoder was tested and dropped — it over-weighted vocabulary overlap. Retrieval uses a small **additive structural boost** so the vector score still dominates.
- **Index-based entity disambiguation.** When candidates are ambiguous, the LLM picks a *numbered* option and we map the index back to the exact string — so it's structurally impossible to paraphrase an entity name.
- **Facets baked into the payload.** Weapon/armor/spell stat cards are parsed into flat, indexed facets so relational filtering and superlatives are exact.

---

## Tech stack

| Layer | Tech |
|---|---|
| **Backend** | Python 3.12 · FastAPI · LangChain · Uvicorn (SSE streaming) |
| **Frontend** | Next.js 16 · React 19 · TypeScript · Tailwind CSS 4 · TanStack Query |
| **Vector DB** | Qdrant (768-dim, cosine; HNSW + payload indexes) |
| **Embeddings** | `BAAI/bge-base-en-v1.5` (768-dim), query-prefixed |
| **LLM** | Google **Gemma 4** (primary) with NVIDIA NIM cross-provider fallback |
| **Persistence** | Supabase Postgres (optional — conversation history) |
| **Deploy** | Backend → GCP Cloud Run · Frontend → Vercel · Qdrant → Qdrant Cloud |
| **Corpus** | Fextralife Elden Ring wiki (base game + Shadow of the Erdtree) — ~39k chunks |

### The LLM tier chain

```
Gemma 4 (gemma-4-26b-a4b-it)  ──▶  Gemma 4 (31b)  ──▶  NVIDIA NIM (nemotron)
  primary · MoE active-4B,          slower dense         cross-provider net
  ~1s to first token                fallback
```

- The **primary is Gemma 4 26b-a4b** (Mixture-of-Experts, ~4B active params) via **Google AI Studio's free tier** —  generous (1,500 requests/day). 
- The **router** uses the same Gemma-first order with structured JSON output (temperature 0).

---

## Run it locally

You need **[uv](https://docs.astral.sh/uv/)** (Python), **Node 20+**, **Docker** (for Qdrant), and a **free [Google AI Studio API key](https://aistudio.google.com/apikey)** (for Gemma). An NVIDIA NIM key is optional (fallback only).

### 1. Clone & configure

```bash
git clone https://github.com/<you>/elden-path.git
cd elden-path
cp .env.example .env      # then fill in GOOGLE_API_KEY (minimum)
```

### 2. Start Qdrant

```bash
docker compose up -d
```

### 3. Load the corpus — pick ONE (both skip the whole scrape→chunk→embed pipeline)

The full data pipeline (scrape the wiki, chunk, GPU-embed ~39k chunks) takes a while and needs a GPU. **For testing, don't run it** — load the prebuilt corpus instead:

**Qdrant snapshot**
Download the snapshot ([Google Drive](https://drive.google.com/file/d/1HvlmUeuHTYjRY25N2NBZF7gF2jzFEspH/view?usp=sharing)) and restore it into your local Qdrant:

```bash
curl.exe -X POST "http://localhost:6333/collections/elden_ring/snapshots/upload?priority=snapshot" \
  -H "Content-Type: multipart/form-data" \
  -F "snapshot=@elden_ring.snapshot"
```


> The **backend** does download the ~400 MB `bge-base` model on first run to embed *your queries* — this is automatic and runs fine on CPU.

### 4. Start the backend (port 3000)

```bash
cd backend
uv run uvicorn api.main:app --reload --port 3000
```

### 5. Start the frontend (port 3001)

```bash
cd frontend
npm install
npm run dev -- -p 3001
```

Open **http://localhost:3001** and ask away. (`frontend/.env.local` should have `NEXT_PUBLIC_API_URL=http://localhost:3000`.)



---

## Project structure

```
backend/
  api/    FastAPI app · routers · config · auth · db
  core/   router · entity_resolver · retriever · llm · pipeline · generate · prompts
data_pipeline/
  discover · scrape · chunker · embed · upload   (offline corpus build)
frontend/
  app · components (ChatView, Message, Sidebar, VoiceToggle) · lib (SSE + typewriter)
```

---

## Future work

- **YouTube transcript ingestion** — pull transcripts from lore channels (VaatiVidya, etc.) into the corpus so the bot can draw on community lore interpretation, not just the wiki's factual pages. This is the biggest planned expansion of the knowledge base.
- **Real authentication** — auth is currently a dev stub; wire up JWT/Supabase auth (the schema already supports per-user conversations).
- **Inline images** — a second pass to place images *within* the answer at the relevant sentence, not just as a source strip.
- **Multi-tagged entity variants** — richer handling of entities that span several wiki pages (phase-variant bosses, altered armor, etc.).

---

## Notes

- Corpus content belongs to the Fextralife Elden Ring wiki and its contributors; this project is a non-commercial fan/portfolio project. *Elden Ring* is © FromSoftware / Bandai Namco.
- Built as a portfolio project to explore production-grade RAG: structured routing, phonetic entity resolution, facet-based relational retrieval, and multi-provider LLM fallback.
</content>
