# SuperMew

A school assistant a parent talks to over WhatsApp, and the four services behind it.

A parent asks about their child — marks, attendance, what the school's handbook says —
and gets an answer grounded in the school's own records and corpus, or an honest refusal.
Nothing is guessed.

**This file documents the chat backend and its RAG engine.** For how the four services
fit together, who trusts whom, and where a request travels, read
[SERVICES.md](SERVICES.md) first — it is the shorter document and the one that explains
the shape of the system.

| Service | Port | What it owns |
| --- | --- | --- |
| `backend/` | 8000 | The agent, the RAG pipeline, chat sessions, document ingest |
| `records/` | 8100 | The academic records facade: authorisation, terms, report cards |
| `identity/` | 8200 | Accounts, tokens, and WhatsApp parent sign-in |
| `sis/` | 8300 | The school's own student information system and registrar console |

Two frontends: `frontend/` is the parent-facing chat UI (Vue 3), and `sis/frontend/` is
the registrar console (React), built into `sis/web/` and served by `sis` at `/ui`.

[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/icey1287/SuperMew)

## Local Deployment

### 1) Prerequisites
- Python `3.12+`
- Recommended package manager: `uv` (`pip` is also supported)
- Docker / Docker Compose (used to start the Milvus dependencies)

### 2) Install dependencies via pyproject
Run from the project root:

```bash
# Option A: recommended (uv)
uv sync

# Run the service
uv run python backend/app.py
# or
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
# or 
.venv\Scripts\uvicorn.exe backend.app:app --host 0.0.0.0 --port 8000 --reload
```

```bash
# Option B: pip
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# Run the service
python backend/app.py
# or
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3) Create the `.env` file

```bash
cp .env.example .env
```

Edit the API keys, model names, and connection URLs in `.env` as needed; see the comments in `.env.example` for details on each variable.

### 4) Docker deployment (database + cache + vector store)
This repo's `docker-compose.yml` bundles both the application dependencies and the Milvus dependencies:
- Application dependencies: `postgres`, `redis`
- Vector dependencies: `etcd`, `minio`, `standalone`, `attu`

```bash
# Start the vector store dependencies
docker compose up -d

# Check service status
docker compose ps

# View logs (optional)
docker compose logs -f standalone
```

Port reference:
- PostgreSQL: `5432`
- Redis: `6379`
- Milvus: `19530`
- Milvus health check: `9091`
- MinIO API: `9000`
- MinIO Console: `9001`
- Attu: `8080`

### 5) Build the frontend (required on first run and after any frontend changes)
On first run, or after modifying the frontend code, you need to install dependencies and build the frontend so the `frontend/dist` directory (served by the backend) gets generated:

```bash
cd frontend

# Install frontend dependencies
npm install

# Build the static bundle
npm run build
```

Once the build finishes, the output is saved automatically under `frontend/dist/`; the backend mounts this directory automatically on startup.

### 6) Start the estate and access it

Start the infra containers first (step 3 above), then each service in its own terminal.
None of them takes an environment variable on the command line — every service reads
`.env` for itself, so there is one place a setting can be wrong:

```bash
# Database migrations for the SIS, once per schema change
uv run alembic -c sis/alembic.ini upgrade head

uv run uvicorn identity.app:app --port 8200
uv run uvicorn records.app:app  --port 8100
uv run uvicorn sis.app:app      --port 8300
uv run uvicorn backend.app:app  --host 0.0.0.0 --port 8000 --reload
```

If a service exits complaining that its port is taken, a previous run is still holding it.
Stop that process rather than moving the port: a stale process answering with stale
settings reads exactly like an edit to `.env` being ignored.

Open in a browser:

| | |
| --- | --- |
| Parent chat UI | `http://localhost:3000` (dev server) or `http://127.0.0.1:8000/` (built) |
| Chat API docs | `http://127.0.0.1:8000/docs` |
| Registrar console | `http://localhost:8300/ui` |
| SIS / records / identity docs | `:8300/docs`, `:8100/docs`, `:8200/docs` |

### 7) Frontend development & debugging (optional)
The frontend is built with Vite + Vue 3. To develop and debug the frontend code:

```bash
cd frontend

# 1. Start the local dev server (runs at http://localhost:3000, with a built-in reverse proxy to the FastAPI backend on port 8000)
npm run dev

# 2. Build the production bundle (output goes to frontend/dist/, served statically by the backend)
npm run build
```

## Project Overview
- **Core capabilities**:
  - LangChain Agent + custom tools.
  - Uploaded documents go through three-tier sliding-window chunking; leaf chunks are embedded and written to Milvus, and parent chunks are written to PostgreSQL.
  - User registration/login, JWT authentication, role-based RBAC access control (admin/user).
  - Session memory and summarization; chat and history persist to PostgreSQL, with Redis caching hot sessions and parent documents.
- **Deployment shape**: four independent FastAPI services + two frontends (Vue 3 for parents, React for registrars) + PostgreSQL, Redis and Milvus.

## Key Innovations
- **Hybrid retrieval in production**: dense vectors + BM25 sparse vectors, Milvus Hybrid Search + RRF ranking, balancing semantic and lexical matching.
- **Low-latency complexity planning with parallel Sub-Agents**: obviously simple, single-fact questions go straight to retrieval via local rules; everything else has FAST_MODEL determine complexity and plan 2-4 sub-questions in a single call. Complex questions run each sub-question's "retrieve -> grade evidence" step in parallel via LangGraph's `Send`, then get deduplicated and merged at the Synthesis node.
- **Corrective RAG with single-choice query rewriting**: after retrieval, a dedicated GRADE_MODEL judges evidence relevance, answerability, and ambiguity via structured output. When evidence is insufficient, FAST_MODEL picks either Step-back or HyDE in a single structured call, and only that one rewrite-retrieval-and-regrade cycle runs.
- **Jina Rerank integration**: API-level reranking after Hybrid/Dense recall, returning `rerank_score` for frontend visualization.
- **Two-way degradation**: automatically falls back to pure dense retrieval if sparse vector generation or the Hybrid call fails, improving robustness.
- **Streaming output**: the backend streams tokens via `agent.astream(stream_mode="messages")`; the frontend uses SSE + ReadableStream for a typewriter effect.
- **Real-time RAG process visualization**: retrieval progress starts showing while the model is still "thinking," powered by an `asyncio.Queue` + background-task architecture that pushes updates in real time during tool execution.
- **Answer cancellation**: the frontend's `AbortController` plus the backend's `StreamingResponse` let users interrupt an in-progress answer at any time.
- **Session summary memory**: old messages are automatically summarized and injected into the system prompt, preserving context while controlling token usage.
- **Document ingestion pipeline**: upload -> chunk -> generate dense/sparse vectors together -> write to Milvus, with automatic cleanup of old chunks on re-upload.
- **Milvus 2.5+ native BM25 hybrid retrieval**: entirely drops the tedious pattern of hand-rolled client-side BM25 serialization and statistics syncing. By binding a `FunctionType.BM25` function to the `text` field in the Milvus collection schema, the vector database extracts sparse features natively on the server side, guaranteeing efficient Dense + Sparse hybrid retrieval with perfectly aligned statistics.
- **Three-tier chunking + Auto-merging**: three-tier sliding-window splitting (L1/L2/L3); retrieval prioritizes L3 recall and automatically merges up to the parent chunk (L3->L2->L1) once a threshold is met.
- **Leaf-only vector storage**: only leaf chunks are written to Milvus, while parent chunks go to the DocStore, reducing vector redundancy while preserving the ability to aggregate context.
- **Extensible tools**: knowledge-base retrieval, figure reading, and a student-records tool that reaches the records facade, all bound per domain profile — a deployment declares the tool names it wants and the registry resolves them.
- **Observable RAG process**: retrieval, grading, rewriting, and source information are all logged; the frontend can expand to inspect every step's details.
- **Query rewriting system**: when evidence is insufficient, FAST_MODEL picks a single method (Step-back or HyDE) and runs only one secondary retrieval, keeping model-call counts and worst-case latency in check.
- **Relevance-score gating**: structured-output `grade_documents` determines whether a rewrite-and-retry retrieval is needed.
- **Real-time thinking-trace display**: using an `asyncio` event-loop pass-through technique, the Agent pushes thinking steps (Searching -> Grading -> Rewriting) to the frontend in real time while running synchronous tools like RAG, grading, and rewriting — completely solving the "silent thinking" problem.

## Roadmap (Todo Lists)

### RAG

#### Data layer & chunking

1. Start with document-structure parsing for a coarse split along structural boundaries, falling back to recursive character chunking to ensure topical units aren't broken apart (2000-3000 tokens); then apply semantic chunking for finer-grained splitting, keeping each chunk within 512-1024 tokens.
2. Special handling for code blocks, tables, and images.
3. Implement a ParentDocument/Auto-merging Retriever strategy --done

#### Recall layer

1. Add parameter sweeps for BM25's k1 and b.
2. Add BM25/dense weighting on top of RRF, tunable via A/B testing.
3. Build a small labeled set to compare gold chunks across dense-only, sparse-only, hybrid, and hybrid + rerank.

#### Generation layer

1. Sub-question decomposition (CoT, a dedicated small decomposition model, deciding how many sub-questions to split into).
2. Multi-document refine (single concatenation vs. sequential refine).
3. Multi-document conflict handling (document A says X, document B says not-X) — explicitly surface "conflicting sources" in the answer.

#### Other

1. Vector embeddings: add multimodal embedding support.
2. Build out a RAG evaluation framework.
3. Rerank strategy evaluation (top_k, candidate_k, recall/rerank ratio).

### Other capability expansion

1. Build a SQL assistant Skill.
2. Implement a pause feature and human-in-the-loop mechanism --done
3. Add question-type classification so simple questions can skip the complex processing pipeline.
4. Expand web search capability.
5. Support multi-step planning with parallel task execution.
6. Build a router node so the LLM can autonomously decide the next action.
7. Improve memory management: integrate solutions like MemO, LangMem.
8. Multi-agent: with too many tools in one place, split tools across specialized agents with clear responsibilities to improve tool-selection accuracy and overall stability.
9. Allow renaming session titles in chat history.
10. Infinite-loop detection and recovery: `_is_stuck` + `attempt_loop_recovery`.

### Backend Service Buildout (completed in this round)

1. Account and permission system
- Added registration/login endpoints: `/auth/register`, `/auth/login`.
- Added a user-info endpoint: `/auth/me`.
- Introduced JWT authentication middleware: requests identify the current user via a Bearer token.
- Permission isolation:
  - `admin`: can upload, delete, and list documents.
  - `user`: can only chat, and query/delete their own session history.

2. Database modeling and persistence migration
- Established core models with SQLAlchemy: `User`, `ChatSession`, `ChatMessage`, `ParentChunk`.
- Migrated chat history from local JSON to PostgreSQL.
- Migrated parent chunk documents (L1/L2) from local JSON to PostgreSQL.

3. Redis caching strategy
- Session message cache: message lists cached per `user + session`.
- Session list cache: session summary lists cached per `user`.
- Parent document cache: parent chunk content cached by `chunk_id`.
- Cache invalidation runs after writes/deletes to maintain consistency.

4. Password security and compatibility
- Newly registered users have their password hashed with PBKDF2-SHA256 (avoiding bcrypt backend compatibility issues).
- Login validation remains compatible with legacy bcrypt hashes, enabling a smooth migration.

## Directory Layout & Architecture
- Backend: `backend/` (a layered package structure, all imports go through `from backend.xxx import`)
  - [app.py](backend/app.py): FastAPI entry point, CORS, static asset mounting.
  - `api/`: HTTP layer
    - [router.py](backend/api/router.py): route aggregation.
    - `routes/`: split into `auth`, `sessions`, `chat`, `documents` files.
    - [resources.py](backend/api/resources.py): shared resources such as Milvus and the upload directory.
  - `chat/`: conversation domain
    - [service.py](backend/chat/service.py): non-streaming / streaming chat entry points.
    - [runtime.py](backend/chat/runtime.py): model clients and per-request Agent creation.
    - [request_context.py](backend/chat/request_context.py): per-request RAG step, RAG trace, and tool-budget context.
    - [storage.py](backend/chat/storage.py): session storage in PostgreSQL + Redis.
  - `rag/`: retrieval augmentation
    - [pipeline.py](backend/rag/pipeline.py): the LangGraph RAG workflow.
    - [utils.py](backend/rag/utils.py): hybrid retrieval, reranking, auto-merging.
  - `indexing/`: document ingestion and vectors
    - [embedding.py](backend/indexing/embedding.py): dense + BM25 sparse vectors.
    - [document_loader.py](backend/indexing/document_loader.py): PDF/Word/Excel chunking.
    - [milvus_client.py](backend/indexing/milvus_client.py), [milvus_writer.py](backend/indexing/milvus_writer.py).
    - [parent_chunk_store.py](backend/indexing/parent_chunk_store.py): the parent-chunk DocStore.
  - `tools/`: `@tool`-decorated functions callable by the LangChain Agent. Bound per
    profile through `TOOL_BUILDERS` in [tools/__init__.py](backend/tools/__init__.py) —
    knowledge-base retrieval, figure reading, and student records.
  - `profiles/`: the domain profile system. A YAML file per deployment declares which
    tools are bound, which RAG rungs run, and what the assistant is called. `school` is
    the profile this deployment runs; see [registry.py](backend/profiles/registry.py).
  - `infra/`: [database.py](backend/infra/database.py), [cache.py](backend/infra/cache.py), [auth.py](backend/infra/auth.py).
  - `db/`: [models.py](backend/db/models.py): ORM models.
  - `schemas/`: Pydantic request/response schemas (chat / documents).
  - `jobs/`: [upload_jobs.py](backend/jobs/upload_jobs.py): async upload/delete job progress.
- The other three services, each deployable on its own and documented in its own README:
  - `identity/` — accounts, JWT signing and verification, and the WhatsApp sign-in flow
    that lets a parent authenticate without a password. See [identity/README.md](identity/README.md).
  - `records/` — the academic records facade. Owns guardian authorisation, terms, report
    card snapshots and the access audit; owns no grades. Reads the system of record
    through `LmsAdapter`. See [records/README.md](records/README.md).
  - `sis/` — the school's own student information system: roster, structure, attendance,
    marks and guardians, plus the registrar console. See [sis/README.md](sis/README.md).
  - `scripts/` — estate-level operator tooling: a health check that walks a parent's
    sign-in across the running services, and a WhatsApp webhook simulator. Both talk
    HTTP only and import no service. School provisioning moved into `sis/` (it is
    `sis` code: `python -m sis.schools`).
- Frontend: `frontend/` — the parent-facing chat UI. (The registrar console is a
  separate React app under `sis/frontend/`, built into `sis/web/`.)
  - Built with a modern, engineered stack (Vite + Vue 3 + TypeScript + Pinia + Axios + Sass).
  - **Frontend architecture & state flow**:
    - **Pinia stores**:
      - `stores/auth.ts`: handles JWT auth state, user registration and login, and keeps Bearer-authenticated requests going.
      - `stores/sessions.ts`: handles creating, asynchronously loading, deleting, and switching between multiple chat sessions.
      - `stores/chat.ts`: caches the message stream and drives reactive updates for each RAG execution step.
      - `stores/documents.ts`: renders the knowledge-base document list and polls the API to track async upload job progress.
    - **Fine-grained component design**:
      - `ThinkingTrace.vue` & `RetrievalTraceDetails.vue`: dynamically render sub-Agent/main-Agent thinking state (Searching, Grading, Rewriting, etc.), including merge and recall details for each sub-question.
      - `References.vue`: a collapsible card that shows knowledge-base source info, including RRF rank, rerank semantic score, number of merged leaf chunks, tier, and page number.
      - `UploadSection.vue` & `DocumentSettings.vue`: the admin control panel, polling and stepping through the multi-stage upload state machine.
    - **Streaming unpacking & active cancellation**:
      - `utils/api.ts`: uses the `fetch` API's `response.body.getReader()` to unpack SSE data chunk by chunk at a low level, paired with an `AbortController` wired to the cancel button so the frontend can actively terminate a long-lived connection.
  - Run `npm run dev` inside `frontend/` to start local development (served at http://localhost:3000).
  - Run `npm run build` inside `frontend/` to produce the production build, output to `frontend/dist/`, which the FastAPI backend serves statically without any extra steps.
- Data: `data/`
  - `documents/`: the original uploaded document files.
- Vector store: Milvus (provided by `docker-compose` or a self-hosted service).

## Core Flows

### 1) End-to-end project flow
1. The user submits a question in the frontend, calling `POST /chat/stream` (streaming).
2. FastAPI's `api/routes/chat.py` returns a `StreamingResponse(media_type="text/event-stream")`.
3. The LangChain Agent decides whether to call a tool based on the question type:
  - Knowledge question -> `search_knowledge_base`
  - Figure or diagram -> `view_figure`
4. If the knowledge-base tool is triggered, execution enters `backend/rag/pipeline.py` to run the retrieval workflow, with each stage pushed to the frontend in real time via `ChatRequestContext`.
5. The retrieval results and RAG trace are returned together, and the Agent streams the final answer (pushed token by token).
6. The frontend's ReadableStream parses the SSE chunks and renders them in real time with a typewriter effect.
7. Meanwhile, messages are persisted to PostgreSQL, with Redis caching speeding up replay of historical sessions.

### 2) RAG flow in detail (the core of the system)
1. **Complexity planning**: `classify_complexity`
  - Obviously short, single-fact questions are classified `simple` directly by local rules, without a model call.
  - Everything else has FAST_MODEL make the simple/complex determination in a single call; a `complex` result also returns 2-4 sub-questions in that same call, with no extra decomposition call needed.
2. **Retrieval execution**
  - simple: goes to `retrieve_initial` and runs a single standard retrieval.
  - complex: each sub-question's "retrieve -> grade evidence" step runs in parallel via LangGraph's `Send`, then `synthesis` deduplicates and merges the results.
  - Calls `retrieve_documents`.
  - First runs a Milvus Hybrid retrieval (Dense + Sparse + RRF) filtered to `chunk_level == 3`; the candidate pool size is determined by `RETRIEVAL_CANDIDATE_K` or `RETRIEVAL_CANDIDATE_MULTIPLIER`.
  - Runs Auto-merging (L3->L2->L1) over the full candidate set of leaf chunks, reading parent chunks from the DocStore.
  - Passes the merged fragments through Jina Rerank and truncates to `top_k` (pipeline stage: `recall_merge_rerank`).
3. **Evidence grading & routing**: `grade_documents`
  - GRADE_MODEL outputs relevance, answerability, ambiguity, confidence, and `route` in a single call.
  - Routing only leads to answering, a single rewrite, HITL clarification/scope selection, or ending with no knowledge; a grading failure raises an explicit error rather than falling back to another implementation.
4. **Step-back / HyDE single-choice rewrite**: `rewrite_question`
  - FAST_MODEL picks one method and generates the corresponding content in a single structured call.
  - Step-back: generates a more abstract, "stepped-back" question, combined with the original question into `rewritten_query`.
  - HyDE: generates a hypothetical answer document used only for retrieval, combined with the original question into `rewritten_query`; this document is never used as answer evidence.
5. **Secondary recall**: `retrieve_rewritten`
  - Runs another L3 recall -> Auto-merging -> Rerank pass against `rewritten_query`.
6. **Answer generation**: the Agent combines the context to produce the final answer.
7. **Observable tracing**: returns a `rag_trace`, including
  - Grading results and routing decisions
  - `rewrite_method`, `step_back_question` / `hyde_document`, and `rewritten_query`
  - Initial/secondary retrieval results
  - Three-tier retrieval and merge info (`leaf_retrieve_level`, `auto_merge_*`)
  - Retrieval score `score` and rerank score `rerank_score`

### 3) Document ingestion flow
1. The frontend uploads a PDF/Word file to `POST /documents/upload`.
2. If a file with the same name already exists: old vectors and parent chunks are first cleared from PostgreSQL and the Redis cache to keep the store consistent.
3. `document_loader.py` runs the three-tier sliding-window chunking and writes the hierarchy metadata (chunk_id / parent_chunk_id / root_chunk_id / chunk_level).
4. L1/L2 parent chunks are written to `parent_chunk_store.py` (DocStore / PostgreSQL).
5. L3 leaf chunks get dense vectors injected via `milvus_writer` (produced locally by `embedding.py`'s `HuggingFaceEmbeddings`), and the raw text is written to the `text` field, which is configured with a native Chinese tokenizing analyzer.
6. Milvus automatically and synchronously triggers native BM25 extraction on the server side, dynamically generating and storing sparse vectors in `sparse_embedding` — no client-side statistics tracking required.
7. Subsequent retrievals can immediately use the new document during recall.

### 4) Milvus 2.5+ native BM25 processing
- **Mechanism**: the project uses Milvus 2.5+'s new built-in full-text search mechanism. When the collection is created, a `FunctionType.BM25` function is defined with `text` as its input field and `sparse_embedding` as its output field.
- **Automatic alignment**: whenever a new text chunk is inserted or deleted, Milvus automatically tokenizes, tallies statistics, and computes the sparse feature vector on the server side. This delivers efficient dense + sparse dual-tower retrieval with zero client-side statistics burden.

### 5) Session memory flow
1. Each turn is written to PostgreSQL keyed by the logged-in user + `session_id`.
2. When the message history grows too long, summary compression kicks in to preserve long-term context.
3. Redis caches the session list and session messages to reduce load on the database from frequent reads.
4. The frontend can read and delete the current user's own conversation history via the session API.

## Tech Stack
- Backend: FastAPI, LangChain Agents, Pydantic, Uvicorn, SQLAlchemy, PostgreSQL, Redis.
- Vector & retrieval: Milvus (HNSW dense index + SPARSE_INVERTED_INDEX sparse index), RRF fusion, Jina Rerank.
- Embeddings & sparse vectors: local dense vectors via `langchain_huggingface` (defaults to `BAAI/bge-m3`); Milvus 2.5+'s native Chinese analyzer and native BM25 feature extraction.
- Frontend: Vite + Vue 3 (SFC) + TypeScript + Pinia + Axios + Marked + Highlight.js + FontAwesome, with an engineered build and static file hosting.
- Toolchain: dotenv config, requests, langchain_text_splitters, langchain_community.loaders.

## Environment Variables
Configure these at the repo root or in your runtime environment:
- Model-related: `ARK_API_KEY`, `MODEL`, `FAST_MODEL`, `GRADE_MODEL`, `BASE_URL`. `FAST_MODEL` handles complexity planning and the Step-back / HyDE single-choice rewrite; `GRADE_MODEL` is dedicated to evidence grading. Both are explicitly required and never substitute for each other or fall back to `MODEL`.
- Dense vectors: `EMBEDDING_MODEL`, `EMBEDDING_DEVICE`, `DENSE_EMBEDDING_DIM` (must match the `dense_embedding` field dimension in the Milvus collection)
- Dense and sparse: dense vectors come from the local embedding model; sparse vectors are automatically generated and maintained by Milvus's Chinese analyzer and BM25 Function
- Rerank-related: `RERANK_MODEL`, `RERANK_BINDING_HOST`, `RERANK_API_KEY`
- Milvus: `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_COLLECTION`
- Database/cache: `DATABASE_URL`, `REDIS_URL`
- Auth-related: `JWT_SECRET_KEY`, `ADMIN_INVITE_CODE`, `JWT_ALGORITHM`, `JWT_EXPIRE_MINUTES`
- Password parameters: `PASSWORD_PBKDF2_ROUNDS`
- Retrieval candidate pool: `RETRIEVAL_CANDIDATE_K` (a fixed candidate count, takes priority), `RETRIEVAL_CANDIDATE_MULTIPLIER` (used when K isn't set: `max(top_k x multiplier, top_k)`, default `3`)
- Auto-merging: `AUTO_MERGE_ENABLED`, `AUTO_MERGE_THRESHOLD`, `LEAF_RETRIEVE_LEVEL`

## API Overview
- Auth
  - `POST /auth/register`: registration (supports a regular-user mode and an admin invite-code mode).
  - `POST /auth/login`: login, returns a Bearer token.
  - `GET /auth/me`: fetch the current logged-in user's info.
- Chat
  - `POST /chat`: chat (non-streaming), params `message`, `session_id`.
  - `POST /chat/stream`: chat (streaming SSE), same params, returns `text/event-stream`.
- Sessions (per-user isolation)
  - `GET /sessions`: list the current user's sessions.
  - `GET /sessions/{session_id}`: fetch messages for one of the current user's sessions.
  - `DELETE /sessions/{session_id}`: delete a session belonging to the current user.
- Documents (admin only)
  - `GET /documents`: list ingested documents and their chunk counts.
  - `POST /documents/upload`: upload and embed a PDF/Word/Excel file.
  - `DELETE /documents/{filename}`: delete a document's vector data (first paginates through chunk text by filename to decrement the persisted BM25 statistics, then deletes from Milvus).

## Streaming Output & Real-Time Retrieval — Technical Deep Dive

#### 1. Cross-Thread Event Scheduling
This is a key architectural design that solves the **"a synchronous tool blocks the async event loop"** problem, common in Python async web services mixing CPU-bound/IO-bound tasks.

**The pain point**:
FastAPI runs on a single-threaded asyncio event loop. To avoid blocking the main thread, LangChain typically runs synchronous tools (like `search_knowledge_base`) in a `ThreadPoolExecutor`. But from a worker thread, you can't directly access the main thread's `asyncio.Queue`, and `asyncio.get_event_loop()` usually fails there.

**The solution**:
We use a **"Request Context + Thread-Safe Callback"** pattern:

1.  **Request context creation (Service Layer)**:
    `chat_with_agent_stream()` creates an independent `ChatRequestContext` for each request, holding that request's `output_queue` and a reference to the main event loop.
2.  **Explicit dependency injection (Tool/RAG Layer)**:
    At runtime, `create_agent_for_request(ctx)` creates the agent for this request, and `make_search_knowledge_base(ctx)` creates a dedicated tool that captures that `ctx`. The RAG pipeline's entry point is `run_rag_graph(question, ctx)`.
3.  **Cross-thread dispatch (Worker Thread)**:
    A RAG node calls `ctx.emit_rag_step(...)`, which internally uses the request's saved `loop.call_soon_threadsafe(queue.put_nowait, event)` to deliver the event back to the main loop.
4.  **Isolation guarantee**:
    RAG steps, the RAG trace, and the knowledge-base tool call counter are all stored on the request context object.

```python
# Core code summary
ctx = ChatRequestContext.for_stream(
    user_id=user_id,
    session_id=session_id,
    output_queue=output_queue,
)
agent = create_agent_for_request(ctx)

# Inside a RAG node
ctx.emit_rag_step("🔍", "Searching the knowledge base...", "Initial retrieval")
```

### 2. Hybrid Search — a deep dive
Rather than hand-rolling complex BM25 feature serialization on the client, the project builds an extremely efficient dual-tower retrieval on top of Milvus 2.5+'s native server-side analyzer:

- **Dense pathway**: dense vectors are generated with `langchain_huggingface.HuggingFaceEmbeddings` (defaults to `BAAI/bge-m3`); the dimension is aligned with the collection schema via `DENSE_EMBEDDING_DIM` (default 1024), and vectors can be L2-normalized to pair with Milvus's `IP` metric.
- **Sparse pathway**:
    - When writing a document, only the raw text needs to be written to the `text` field, which has the `chinese` analyzer enabled for tokenization.
    - The Milvus server automatically runs the bound `FunctionType.BM25` function, dynamically generating the corresponding sparse embedding and syncing it into the `sparse_embedding` index — with statistics perfectly aligned.
- **Milvus fusion**:
    - Milvus's `AnnSearchRequest` is used to issue the dense and sparse multi-path retrieval requests concurrently.
    - **RRFRanker (Reciprocal Rank Fusion)**: uses a reciprocal-rank fusion algorithm with `k=60` to merge the two recall result sets without any parameter tuning, avoiding the difficulty of tuning an `alpha` weight in a weighted-sum approach.

### 3. The frontend "Thinking State Machine"
The frontend's `stores/chat.ts`, together with the reactive `ThinkingTrace.vue` component, maintains a small state machine to handle the complex mixed stream coming back over SSE:

1.  **Idle**: waiting for user input.
2.  **Thinking (Initial)**: the request is received, a message bubble is created, and `isThinking` is set to `true`.
3.  **Thinking (Active RAG)**: a `type: "rag_step"` event arrives.
    - The state machine keeps `isThinking=true`.
    - The current RAG step text and status detail card update dynamically (e.g. showing "Rewriting the query...", "Auto-merging complete", etc.).
    - Steps are appended to the message item's `ragSteps` array and rendered by the component in real time.
4.  **Streaming**: the first `type: "content"` event arrives.
    - **Immediate switch**: `isThinking` is set to `false`.
    - The bubble is neither destroyed nor recreated — the thinking-detail header is simply hidden, and Markdown body text starts streaming into the same bubble.
    - This produces a seamless visual transition from "dynamic retrieval-step thinking" to "LLM streaming answer," which reads extremely smoothly.

## Overall Architecture

```
User sends a message
    │
    ▼
POST /chat/stream → StreamingResponse(text/event-stream)
    │
    ▼
chat_with_agent_stream()
    │
    ├── Create the unified output queue (asyncio.Queue)
    ├── Create ChatRequestContext.for_stream(...)
    ├── create_agent_for_request(ctx) binds a tool dedicated to this request
    ├── Start the _agent_worker background task (asyncio.create_task)
    │     └── agent.astream(stream_mode="messages") yields token by token
    │           ├── AIMessageChunk (text) → enqueued as {"type": "content"}
    │           └── tool_call_chunks (tool calls) → skipped
    │
    └── Main loop: await output_queue.get() → yield SSE
          ▲
          │ (concurrently) RAG tools run in the thread pool
          │ ctx.emit_rag_step() → loop.call_soon_threadsafe → enqueued
          │ {"type": "rag_step"} is immediately pulled off the queue and pushed to the frontend
```

### Backend Implementation

#### 1) Streaming generation (`backend/chat/service.py`)
- Uses LangGraph's `agent.astream(stream_mode="messages")` to get `AIMessageChunk` token by token.
- Filters out `tool_call_chunks`, forwarding only text content to the frontend.
- **Key design**: the Agent's streaming loop runs inside an `asyncio.create_task` background task; the main generator is only responsible for pulling events off the unified `output_queue` and yielding them. This lets RAG steps keep streaming to the frontend in real time even while a tool is executing (i.e. while the agent is blocked waiting on a tool's return value).

#### 2) Real-time RAG step pushing (`backend/tools/knowledge.py` + `backend/rag/pipeline.py`)
- `ChatRequestContext.emit_rag_step(icon, label, detail)` uses the `loop.call_soon_threadsafe()` captured when the request context was created to safely push a step from a synchronous thread into this request's async queue.
- `make_search_knowledge_base(ctx)` creates a tool dedicated to this request; the LLM still only sees the `query` parameter, while the Python closure holds the current request's `ctx`.
- `backend/rag/pipeline.py` receives the context via `run_rag_graph(question, ctx)`, grouping sub-question progress under safe labels (e.g. `Sub-question 1`).
- `backend/rag/pipeline.py` emits a step at each key node:
  - `retrieve_initial` → "Searching the knowledge base..."
  - `grade_documents` → "Evaluating document relevance..."
  - `rewrite_question` → "Choosing a Step-back / HyDE rewrite method"
  - `retrieve_rewritten` → re-runs retrieval using the single method chosen this round

#### 3) SSE protocol format
Each event has the shape `data: {JSON}\n\n`, with a `type` field:
- `content`: a text token (typewriter effect)
- `rag_step`: a real-time retrieval step (`{icon, label, detail}`)
- `trace`: the full RAG trace info (sent once the answer is complete)
- `error`: error information
- `[DONE]`: end-of-stream marker

#### 4) StreamingResponse configuration (`backend/api/routes/chat.py`)
```python
StreamingResponse(
    event_generator(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Disable Nginx buffering
    },
)
```

### Frontend Implementation

#### 1) ReadableStream parsing (`utils/api.ts`)
- Reads chunk by chunk with `response.body.getReader()` + `TextDecoder`.
- Manually splits SSE events on `\n\n`, parsing the JSON after the `data: ` prefix.
- `content` events are appended to the message text; `rag_step` events are appended to the retrieval-step array and simultaneously update the thinking-status text.

#### 2) A unified "thinking bubble"
- As soon as a message is sent, a bubble with `isThinking: true` is created, showing bouncing dots plus dynamic text.
- When a `rag_step` arrives, `thinkingLabel` updates to reflect the current step (e.g. "Searching the knowledge base...").
- When the first `content` token arrives, `isThinking` is set to `false`, and the same bubble seamlessly switches to a normal text stream.
- **There are never two separate bubbles** — thinking, retrieval, and the answer all happen inside the same bubble from start to finish.

#### 3) Vue 3 reactivity notes
- Access is via the `this.messages[botMsgIdx]` index (rather than a cached object reference) to make sure you get Vue's reactive proxy.
- The `ragSteps` array triggers reactive updates via `push()`.

### Cancellation

#### Frontend
- The send button switches to a red cancel button while `isLoading` is true (`v-if`/`v-else`).
- Clicking it calls `AbortController.abort()`, canceling the in-flight `fetch` request.
- An `AbortError` is caught, and the bubble displays "(answer cancelled)".

#### Backend
- FastAPI's `StreamingResponse` detects a dropped socket when the client disconnects (e.g. the browser fires `abort()` or the tab is closed).
- Python's generator protocol throws a `GeneratorExit` exception into the response generator.
- **Implementation detail**: uses **proactive defensive programming** — explicitly catching `GeneratorExit` and calling `agent_task.cancel()`.
- **Why not rely on the framework's automatic cancellation?**: although Starlette/FastAPI has a `BaseHTTPMiddleware`-based cascading cancellation mechanism, the cancellation signal can be delayed or lost along the chain under complex background-task structures or certain middleware configurations. Explicitly calling `.cancel()` guarantees **deterministic resource cleanup**.
- **How the immediate cutoff works**: `agent_task.cancel()` immediately injects an `asyncio.CancelledError` at the task's suspension point. For a streaming LLM request, this triggers `httpx` to close the TCP connection. The server side (e.g. OpenAI) detects the dropped client and stops inference right away, achieving **real token savings**.

## Changelog

### 2026-06-12 Full migration to Milvus 2.5+ native BM25 and transactional, reliable deletion
- **Native server-side BM25**: uses Milvus 2.5+'s built-in Chinese tokenizer and BM25 pipeline function; sparse features and statistics are now maintained automatically by the vector store.
- **Automatic schema upgrades**: refined the `ensure_collection` logic to auto-detect an outdated schema and seamlessly drop-and-rebuild it.
- **Transactional one-click deletion**: implemented a highly reliable, strongly consistent `delete_document_transactionally` deletion coordinator that cleans up Milvus vector data, cascading PostgreSQL chunk records, and the Redis hot cache in one shot, avoiding any dangling/orphaned data.
- **Enterprise-grade text sanitization**: upgraded the text-cleaning logic with Unicode NFC normalization and thorough filtering of non-printable/zero-width/lone-surrogate characters (PUA/C0/C1), fixing character-set compatibility errors between PostgreSQL and Milvus.

### 2026-06-12 Refactored the frontend from a single-file CDN page to an engineered Vite + Vue 3 + TS component architecture
- **Modernized architecture**: refactored the previously bloated, all-in-one HTML/CDN page into a standard **Vite + Vue 3 (SFC) + TypeScript + Pinia + Axios + Sass** engineered project, with all components and state highly decoupled.
- **State & routing management**: used Pinia to set up four core stores — `auth`, `sessions`, `chat`, `documents` — sharing core data.
- **Higher-order interactive UI**: added a detailed streaming upload-progress card, automatic card collapse after a successful upload, an elegant collapsible References display, smooth Thinking-bubble transitions, and more.

### 2026-06-03 Adaptive complex-question decomposition, parallel Sub-Agents, and rerank gating
- **Low-latency complexity planning**: obviously simple, single-fact questions go straight to retrieval via local rules; everything else has FAST_MODEL determine complexity in a single call, returning 2-4 sub-questions at the same time when complex.
- **Parallel sub-Agent retrieval**: uses LangGraph's `Send` API to call `rag_sub_agent` in parallel, with each sub-question only running retrieve and grade, avoiding unreachable nested graphs and a second rewrite pass.
- **Clean sub-step grouping**: the frontend was updated to handle the parallel sub-flow, building independent grouping labels for sub-questions in the RAG Step SSE data to avoid interleaved, duplicated groups and visual confusion.
- **Reranking and clear routing**: `RERANK_MIN_SCORE` filters out noise; an empty retrieval ends immediately, and only when there's a relevant signal but insufficient evidence does the single Step-back / HyDE rewrite run.

### 2026-06-02 General RAG capability improvements and backend lifecycle refactor
- **General RAG enhancements**: added long-term session-summary memory (Context Manager Notes), locally truncated titles from the first question, and a nicely collapsible display card for multi-source references.
- **gRPC connection lifecycle optimization**: switched Milvus database client access from a global connection pool to short-lived sessions (a `session()` context manager), establishing a short connection session per request — completely eliminating stale gRPC channel issues caused by long-hanging connections.
- **Backend layering and package-dependency decoupling**: thoroughly refactored the backend code's package structure, removed the re-export mechanism, resolved circular dependencies caused by cross-imports, and standardized environment loading.

### 2026-06-01 Refactored the recall-merge-rerank pipeline
- **Modular pipeline**: refactored the RAG internals into a tightly controlled "recall -> auto-merge -> semantic rerank" pipeline, with unified parameter configuration and multi-tier RAG trace tracking.
- **Preserving high scores during dedup/merge**: fixed the algorithm that aggregates rank scores in-loop while merging L3 -> L2/L1 leaf chunks upward, preventing high-confidence recall scores from being lost during deduplication.

### 2026-03-21 Backend service upgrade (auth + database + cache)
- Added an auth and permissions module: registration, login, JWT, admin access control.
- Migrated chat history from local JSON to PostgreSQL, isolating session data per user.
- Migrated parent-chunk storage from local JSON to PostgreSQL.
- Introduced Redis caching for sessions and parent documents, improving read performance and reducing database load.
- Upgraded the API to be token-driven, removing the legacy pattern where the frontend passed `user_id` directly.
- Restricted document-management endpoints to the admin role, preventing regular users from accidentally modifying the knowledge base.
- Upgraded the password hashing scheme to PBKDF2-SHA256, while remaining compatible with legacy bcrypt verification.

### 2026-03-13 Three-tier chunking and Auto-merging upgrade
- Added three-tier sliding-window chunking (L1/L2/L3), writing hierarchy metadata onto each chunk.
- Adjusted the storage strategy to leaf-only: only L3 leaf chunks are written to Milvus, with L1/L2 written to the local DocStore.
- Auto-merging now pulls parent chunks from the DocStore, reducing redundant vector storage.
- Added three-tier retrieval and auto-merge step events to the thinking trace.
- Added `leaf_retrieve_level` and `auto_merge_*` fields to `rag_trace`, and these fields are also preserved when reading historical sessions.

### 2026-02-19 Fixed the real-time RAG thinking trace
- **Problem**: because the Agent ran synchronous tools (like `search_knowledge_base`) inside a thread pool, it couldn't correctly access the main thread's asyncio event loop, causing `emit_rag_step` events to be dropped and leaving the frontend's "thinking" bubble frozen.
- **Fix**:
  1. **Backend (`service.py`)**: create a `ChatRequestContext` per request, capturing the main thread's `loop` and this request's `output_queue`.
  2. **Backend (`backend/tools/knowledge.py` + `backend/rag/pipeline.py`)**: use a per-request tool factory and an explicit `ctx` parameter to dispatch RAG steps across threads, avoiding cross-request mix-ups.
  3. **Frontend (`stores/chat.ts`)**: initialize an empty `ragSteps: []` array when sending a message, so Vue's reactivity system can immediately track subsequent `push()` calls.
- **Result**: after a user asks a question, the thinking bubble now shows retrieval steps updating live (e.g. "🔍 Searching the knowledge base..." -> "📊 Evaluating document relevance..."), instead of a static "Thinking...".
