# RAG over Kubernetes Docs

Hybrid (BM25 + pgvector) semantic RAG with a pre-pipeline LLM safety guard. Fully local: Postgres/ParadeDB + Ollama + sentence-transformers reranker + Streamlit. No API keys.

> Deeper, code-level documentation (architecture, data model, operations, testing): see [`openwiki/`](openwiki/quickstart.md).

## Stack

| Layer | Tool |
|---|---|
| Database + Index | ParadeDB (pgvector + BM25) |
| Embeddings | `nomic-embed-text-v2-moe` via Ollama |
| Semantic Chunker | LlamaIndex `SemanticSplitterNodeParser` |
| Hybrid Fusion | RRF (Reciprocal Rank Fusion, k=60) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | `minimax-m3:cloud` via Ollama (served by Ollama locally; the `:cloud` tag is just the model name) |
| Pre-pipeline Guard | `llama-guard3:1b` (fail-closed, safe/unsafe) |
| Eval | RAGAS (faithfulness, answer relevancy, context precision) |
| UI | Streamlit |

## Architecture

### V1 — Hybrid retrieval

The baseline: dense (pgvector) + sparse (BM25) retrieval fused by RRF, reranked, then answered. No safety gate yet — every query reaches retrieval and the LLM.

![V1 architecture](https://github.com/vishal24p/learning/raw/main/rag/assets/version1-arch.webp)

### V2 — Guard prepended

A pre-pipeline LLM safety guard (`llama-guard3:1b`) sits in front of V1. Only `safe` queries reach retrieval; `unsafe` queries halt before any embedding, Postgres query, or generation.

![V2 architecture](https://github.com/vishal24p/learning/raw/main/rag/assets/version2-arch.webp)

## Version log

### V1 -- Hybrid retrieval, oversized chunks

<img src="https://github.com/vishal24p/learning/raw/main/rag/assets/version1.gif" alt="V1 demo" width="100%">

> [version1.gif](https://github.com/vishal24p/learning/blob/main/rag/assets/version1.gif) for the animated loop.

Chunker: SemanticSplitterNodeParser, buffer 2, breakpoint percentile 95. Tight window plus strict threshold meant splits almost never fired on docs where paragraphs only loosely related. Whole pages stayed glued. Average chunk size: **2,813 chars**.

Retrieval: RRF pool of 20 → cross-encoder rerank → top 5 to LLM.

```
Faithfulness      1.000
Answer Relevancy  0.386
Context Precision 0.333
```

Faithfulness at 1.0. Nothing fabricated; everything came from retrieved context. The 0.386 answer relevancy is the real failure. On "what is the use of Kubernetes?" retrieval pulled one Overview chunk covering orchestration, scaling, networking, deployments--all in one block. The model summarized it faithfully, but it never answered the question.

The reranker scored that chunk **-0.44**. Negative. It knew the chunk was off. Still landed in the final 5 because nothing better existed upstream.

**Diagnosis**: 2,813-character chunks bury the two sentences that actually answer the question inside twelve other topics. Dense retrieval can't isolate them. The reranker can't fix what the chunker broke.

### V1.1 -- Re-chunk + positive filter

<img src="https://github.com/vishal24p/learning/raw/main/rag/assets/version2.gif" alt="V2 demo" width="100%">

> [version2.gif](https://github.com/vishal24p/learning/blob/main/rag/assets/version2.gif) for the animated loop. (On disk the V1.1 demo is named `version2.gif`; the V1 demo is `version1.gif`.)

Chunker retuned to buffer 5, breakpoint percentile 80. Same corpus, same embed model, fresh database. 14,734 chunks. Average **1,045 chars**.

`filter_positive_chunks()` added post-rerank: anything below reranker score zero drops out before the LLM sees it. If every candidate is negative, keep the single best so the model still has grounding.

```
V1                              V1.1
Faithfulness      1.000        Faithfulness      0.889
Answer Relevancy  0.386        Answer Relevancy  0.661
Context Precision 0.333        Context Precision 1.000
```

Context precision 0.333 → 1.000. Every chunk in the context window is now relevant. Answer relevancy nearly doubles. Faithfulness drops from 1.000 → 0.889 because the model starts going beyond retrieved context when chunks are tight enough to answer from. Worth the trade-off; answers now answer the question.

### V2 -- Pre-pipeline LLM safety guard

A separate small model call (`llama-guard3:1b`) runs **before** any retrieval. Classifies `safe` or `unsafe`. On `unsafe` the pipeline halts: no embedding call, no Postgres query, no generator call, no RAGAS button. The refusal is logged with the Llama Guard category code (S1..S13) for audit.

Guard is gated by `GUARD_ENABLED=true` in `.env`. Flip to `false` and the pipeline runs unguarded, an A/B switch against baseline. Retrieval and generation logic from V1.1 carry over unchanged.

## Run it

### Prerequisites

- **Python 3.11+** in a virtualenv (`python -m venv .venv`).
  - No dependency manifest is committed; install the packages the code imports
    (`ollama`-py, `psycopg`, `llama-index`, `sentence-transformers`, `ragas`,
    `streamlit`, `langchain-ollama`). See [`openwiki/operations.md`](openwiki/operations.md).
- **Ollama** running on `http://localhost:11434`, with these models pulled:
  - `ollama pull nomic-embed-text-v2-moe` (embeddings)
  - `ollama pull llama-guard3:1b` (pre-pipeline guard)
  - `ollama pull minimax-m3:cloud` (generator)
  - `ollama pull gemma4:e2b` (RAGAS judge; only needed for evaluation)
- **ParadeDB** (Postgres with the `vector` and `pg_search` extensions), listening
  on the port set in `.env` (default `56432`). This repo ships no container
  launcher script in git, so stand up ParadeDB yourself (Docker image
  `paradedb/paradedb`) and ensure the extensions are available.
- **Corpus**: the Kubernetes docs mirror under `website-main/content/en/docs`.
  This directory is gitignored (regenerated from upstream); clone the
  `kubernetes/website` repo there, or point `scripts/reindex_v2.py --corpus` at
  your own markdown tree.
- **`.env`**: copy `.env.example` to `.env` and set `DB_PASSWORD` (and adjust
  `DB_*` / model names as needed). `.env` is gitignored.

### Setup & launch

```bash
# 1. Create + migrate the schema (extensions, chunks table, BM25 + HNSW indexes).
python scripts/apply_migration.py

# 2. Build chunks_v2 from the corpus (creates rag_v2 DB if missing).
python scripts/reindex_v2.py

# 3. Open the Streamlit UI (sets PYTHONPATH, launches http://localhost:8501).
run_ui.cmd          # Windows
#   or: python scripts/run_ui.py
```

CLI smoke tests (no UI) live under `scripts/`: `run_pipeline.py`,
`run_pipeline_with_rerank.py`, `run_rag.py`. Evaluation: `run_ragas.py`.
