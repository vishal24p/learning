# Architecture Review — RAG over Kubernetes Docs

**Scope:** `C:\Users\visha\rag\rag` (the `rag/` project).  
**Reviewed on:** 2026-07-16.  
**Current branch:** `main`, latest commit `8c38b2a`.

---

## 1. Executive Summary

The project is a locally-run, hybrid Retrieval-Augmented Generation (RAG) pipeline for Kubernetes documentation. It couples a ParadeDB-backed vector + BM25 search with local Ollama models (embedding, guard, generator), a cross-encoder reranker, and RAGAS evaluation behind a Streamlit UI. The codebase is well-organized for a learning/portfolio project: each pipeline stage has its own module, environment config is centralized, and the code is extensively documented inline.

From a strict architectural-patterns perspective, the codebase follows a **functional / pipeline decomposition** rather than a layered Clean/Hexagonal/DDD structure. Business rules are embedded in the pipeline modules themselves, and infrastructure concerns (Postgres calls, Ollama clients, reranker model loading) leak directly into retrieval and generation code. That is appropriate for the current scope and team size, but it will become a scaling bottleneck if the project moves toward multi-tenant serving, testing in CI, or swapping out models/databases.

**Verdict:** Solid, readable, demonstrable v2 prototype. The next round of evolution should focus on (a) decoupling from global config, (b) introducing repository/adapter interfaces so tests need no real Postgres/Ollama, and (c) collapsing duplicated entry-point wiring in `scripts/` and `scripts/streamlit_app.py`.

---

## 2. High-Level Component Map

```text
                                     [User query]
                                           |
                              src.guard.pipeline.guard_or_terminate
                              (llama-guard3:1b via Ollama; fail-closed)
                                           |
                                           v
                         +--------------------------------+
                         |  src.retrieval.hybrid_retriever |
                         |   dense (pgvector cosine)       |
                         |   sparse (ParadeDB BM25)        |
                         |   fused via RRF k=60            |
                         +--------------------------------+
                                           |
                                           v
                         +--------------------------------+
                         | src.retrieval.reranked_retriever |
                         |   -> CrossEncoderReranker        |
                         |   -> filter_positive_chunks     |
                         +--------------------------------+
                                           |
                                           v
                         +--------------------------------+
                         |  src.generation.generator       |
                         |   system + user prompt with [n]   |
                         |   Ollama chat / complete fallback|
                         +--------------------------------+
                                           |
                                           v
                                     [Answer + citations]
```

**Sidecar modules:**

- `src.config.settings` — environment-driven dataclass (database, Ollama models, reranker params).
- `src.llm.factory` — cached constructors for Ollama embedding/LLM clients.
- `src.chunking.semantic_chunker` + `src.loaders.markdown_loader` + `src.embeddings.embedder` + `src.storage.postgres_store` — ingestion side.
- `src.evaluation.*` — RAGAS scoring wrapper.
- `scripts/` — CLI smoke tests, Streamlit launcher, migration helper, reindex helper.

**Storage schema:** `src/db/migrations/001_init.sql` creates a single `chunks` table with `pgvector` HNSW index and `pg_search` BM25 index. A parallel `rag_v2/chunks_v2` workflow exists for experimenting with different chunker parameters without wiping v1 data.

---

## 3. What is Done Well

### 3.1 Clear pipeline decomposition

Every stage of the RAG pipeline has a dedicated, small module:

| Stage | Module | Responsibility |
|---|---|---|
| Guard | `src/guard/{guard,pipeline,reason,schema}.py` | Pre-pipeline safety classification, fail-closed, audit logging |
| Dense retrieval | `src/retrieval/dense_retriever.py` | pgvector cosine search |
| Sparse retrieval | `src/retrieval/sparse_retriever.py` | ParadeDB BM25 search |
| Fusion | `src/retrieval/hybrid_retriever.py` | RRF over dense + sparse ranks |
| Reranking | `src/retrieval/{reranker,reranked_retriever}.py` | CrossEncoder scoring + positive filtering |
| Generation | `src/generation/{generator,prompts}.py` | Prompt construction, LLM call |
| Ingestion | `src/ingestion/index.py` | Load -> chunk -> embed -> store |

This makes the code easy to navigate and the README/diagrams are honest about the data flow.

### 3.2 Environment-driven, centralized configuration

`src/config.py` is the single source of truth for `.env` variables. It uses a frozen dataclass and validates boolean flags explicitly. This avoids the common anti-pattern of `os.getenv` calls scattered through business logic.

### 3.3 Thoughtful operational fallbacks

- `CrossEncoderReranker` can run with an identity fallback if the model is unavailable.
- `Generator` falls back from chat API to `complete()` if the model rejects chat formatting.
- `filter_positive_chunks` keeps at least one chunk when all scores are negative, preventing silent refusals.

### 3.4 Fail-closed guard with auditability

The pre-pipeline guard (`llama-guard3:1b`) is genuinely fail-closed: empty responses, free text, exceptions, and malformed "safe" outputs all terminate. Every verdict emits a structured `guard_decision` log line. `GUARD_ENABLED` is a single env-var A/B switch.

### 3.5 Versioned reindexing workflow

`scripts/reindex_v2.py` lets the developer iterate on chunker settings (buffer size / breakpoint percentile) and write to a separate `rag_v2.chunks_v2` table/database, avoiding destructive experiments. This is good experimental hygiene.

### 3.6 Test-friendly seams in some modules

`HybridRetriever`, `CrossEncoderReranker`, and `LlamaGuard3` accept constructor-injected collaborators, which the existing tests exploit with fakes. That pattern should be expanded.

---

## 4. Architectural Gaps and Risks

### 4.1 Global config is imported by almost every module

Nearly every module imports `from src.config import settings` at the top level. Consequences:

- Unit tests must monkeypatch and `importlib.reload` modules to flip flags (e.g., `GUARD_ENABLED`).
- Code cannot be instantiated with different configs in the same process.
- Swapping to a different config source (YAML, Vault, CLI args) requires touching every file.

**Recommendation:** Pass configuration into constructors / function arguments. Keep `src.config.settings` only as the default bootstrap object, not as a global dependency.

### 4.2 No abstract ports/adapters — infrastructure is wired directly

Under a Clean/Hexagonal lens, the inner retrieval/generation logic should not know about psycopg, Ollama, or sentence-transformers. Today:

- `DenseRetriever` constructs SQL inline and calls `psycopg.connect`.
- `SparseRetriever` depends on ParadeDB's `paradedb.score()` and `@@@` operator.
- `Embedder` directly imports the LlamaIndex Ollama embedding class.
- `Generator` directly calls `build_llm()` from the infrastructure factory.

This makes it impossible to unit-test retrieval without a running Postgres or to test generation without Ollama.

**Recommendation:** Introduce small abstract interfaces (ports) for `ChunkStore`, `EmbedderPort`, and `LLMPort`. The existing Postgres/Ollama/LlamaIndex code becomes adapters behind those interfaces. Use in-memory fakes in unit tests.

### 4.3 Entry points duplicate pipeline wiring

- `scripts/run_rag.py`
- `scripts/run_pipeline_with_rerank.py`
- `scripts/streamlit_app.py`

Each of these manually assembles `HybridRetriever`, `RerankedRetriever`, `Generator`, `guard_or_terminate`, and `filter_positive_chunks`. Streamlit additionally duplicates the flow in a non-trivial UI-specific way. Any change to pipeline behavior (e.g., adding a new filter) must be made in multiple places and can drift out of sync.

**Recommendation:** Create a single `RAGPipeline` orchestrator class or function that accepts configuration and returns the answer. CLI and UI should only parse input/output.

### 4.4 RerankedRetriever is called twice in the Streamlit UI

`streamlit_app.py` calls `hybrid.retrieve(...)` to compute the RRF pool, then calls `reranked.retrieve(...)` which internally runs the hybrid retriever *again* for the same query. This embeds the query and hits Postgres twice per user submission.

**Recommendation:** Re-use the first retrieval result, or change `RerankedRetriever` to accept pre-computed candidates.

### 4.5 Data model is implicit and string-keyed

Chunks/retrieval results are `dict` objects with fields like `chunk_id`, `content`, and `score`. There is no typed `Chunk`, `RetrievedChunk`, or `Answer` dataclass. This is error-prone and makes refactoring harder.

**Recommendation:** Add small dataclasses for domain concepts (`Chunk`, `RetrievedChunk`, `GuardDecision` already exists and is good, do the same for pipeline data).

### 4.6 `PostgresStore` mixes schema knowledge with raw SQL

The store knows the exact column list and does string substitution for table names. While the table name is validated with a regex, the SQL is still assembled via f-strings rather than an ORM or a query builder. The migration is also mutated by string replacement in `apply_migration.py` and `reindex_v2.py`.

**Recommendation:** Keep DDL in plain SQL files and load them as templates. Use SQL placeholders for identifiers only after validation; ideally migrate to a lightweight migration tool (e.g., `yoyo-migrations` or `alembic` if an ORM is adopted).

### 4.7 Evaluation code instantiates the generator directly

`src/evaluation/ragas_runner.run_rag_pipeline` imports `Generator()` and calls it, so RAGAS evaluation cannot be exercised with a fake responder. This couples metrics to live Ollama/Postgres.

**Recommendation:** Make `run_rag_pipeline` accept a `RetrieverPort` and `GeneratorPort` so evaluation runs against test doubles when needed.

### 4.8 Package-level imports are sometimes circular-risky

`src/guard/schema.py` imports from `src/guard/guard` only to re-export string constants. This is a minor circular-import risk if `guard.py` ever needs to reference schema-level types. Today it is fine, but worth watching.

---

## 5. Recommendations by Priority

### Immediate (low effort, high value)

1. **Introduce a typed `RAGPipeline` orchestrator** that wraps guard -> retrieve -> rerank -> generate. Use it from all CLI and UI entry points so wiring logic lives once.
2. **Cache the RRF pool** in the UI path so `hybrid.retrieve` is not invoked twice.
3. **Add dataclasses for pipeline data** (`Chunk`, `RetrievedChunk`, `PipelineResult`) and convert `dict` return values over time.

### Short-term (medium effort)

4. **Remove top-level imports of `src.config.settings`** from library modules where possible. Pass settings/config objects into constructors. This dramatically improves testability.
5. **Define `ChunkStore`, `EmbedderPort`, and `LLMPort` abstractions.** Implement in-memory fakes and add unit tests that need no Ollama or Postgres.
6. **Unify SQL migration handling.** Treat `001_init.sql` as the source of truth and apply string transforms through a dedicated render step rather than inline regex in two scripts.

### Long-term / if productionizing

7. **Introduce a real dependency-injection container** or factory module so retriever/LLM instances are constructed once per process and injected into the pipeline.
8. **Add integration tests with a disposable Postgres container** (e.g., Testcontainers) to validate the SQL paths end-to-end.
9. **Separate the ingestion job from the query service.** Indexing is currently a heavyweight batch script; a production version should probably be a separate deployable or a queued worker.
10. **Monitoring / observability:** the existing logging is good, but add structured request/response spans (timing per stage, guard verdict, retrieval counts) for operational dashboards.

---

## 6. Current Architecture vs. Clean/Hexagonal Architecture

| Aspect | Current State | Clean/Hexagonal Ideal |
|---|---|---|
| **Domain model** | `dict` chunks, no entities | Typed entities/value objects |
| **Use cases** | Inline in scripts and Streamlit | Thin orchestrators, framework-free |
| **Ports** | None explicit | `ChunkStore`, `EmbedderPort`, `LLMPort` |
| **Adapters** | Direct psycopg/Ollama calls | Adapters behind ports |
| **Config** | Global `settings` import | Injected at composition root |
| **Testability** | Partial; needs monkeypatch | In-memory adapters, pure unit tests |
| **Composition** | Manual in each entry point | DI container / factory module |

The project is intentionally not at the ideal state today, and that is appropriate for a portfolio/learning artifact. The table above is meant to clarify where the codebase sits and where to invest if the scope grows.

---

## 7. Risk Heat Map

| Area | Risk Level | Why |
|---|---|---|
| Config coupling | Medium | Tests are harder than they need to be; env defaults can mask misconfiguration. |
| Database coupling | Medium-High | Retrieval cannot be unit-tested without Postgres; SQL strings are brittle. |
| Entry-point duplication | Medium | Behavior can drift between CLI and UI. |
| Streaming/performance | Low-Medium | Double retrieval in UI is wasteful but not dangerous. |
| Guard correctness | Low | Fail-closed semantics and audit logging are sound. |
| Operational deployability | Medium | No container/Dockerfile for the app; ParadeDB + Ollama are external prerequisites. |

---

## 8. Summary Statement

The RAG prototype is a coherent, well-documented system that successfully demonstrates hybrid retrieval, reranking, a pre-pipeline safety guard, and local evaluation. Its current architecture is optimized for clarity and demonstration, not for long-term maintainability or production multi-user serving. The highest-leverage next steps are: (1) a single `RAGPipeline` orchestrator shared by CLI/UI, (2) replacing global `settings` imports with constructor injection, and (3) introducing repository/adapter ports so core logic can be tested without real infrastructure.
