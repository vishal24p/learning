# RAG over Kubernetes Docs

**Hybrid (BM25 + pgvector) Semantic RAG pipeline with a pre-pipeline LLM safety guard.**

Built on Postgres/ParadeDB + Ollama + sentence-transformers reranker + Streamlit. All local. No API keys needed.

## Stack

| Layer | Tool |
|---|---|
| Database + Index | ParadeDB (pgvector + BM25) |
| Embeddings | `nomic-embed-text-v2-moe` via Ollama |
| Semantic Chunker | LlamaIndex `SemanticSplitterNodeParser` |
| Hybrid Fusion | RRF (Reciprocal Rank Fusion, k=60) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | `minimax-m3:cloud` via Ollama |
| Pre-pipeline Guard | `llama-guard3:1b` (fail-closed, safe/unsafe classification) |
| Eval | RAGAS (faithfulness, answer relevancy, context precision) |
| UI | Streamlit |

## Architecture

### V1 -- RRF -> Rerank -> Generate

```
Query -> Dense(pgvector) ---+
                            +-- RRF k=60 -> CrossEncoder -> Generator -> Answer
Query -> Sparse(BM25 @@@) --+
```

### V2 -- Guard added before anything

```
Query -> Llama-Guard3 --safe-> RRF -> Rerank -> Generator -> Answer
                       +unsafe-> Refusal (no embed, no DB, no gen)
```

## Versions

### V1 -- First week with RAGAS

<video src="https://github.com/vishal24p/learning/raw/main/rag/assets/version1.gif" autoplay loop muted playsinline width="100%"></video>

**Chunker**: SemanticSplitterNodeParser. Buffer size 2, breakpoint percentile 95. Window was too tight, threshold too strict. On docs where every paragraph is loosely related to the next, the threshold fired almost never. Whole pages stayed glued. Average chunk: **2,813 characters**.

**Retrieval**: RRF pool of 20 -> cross-encoder rerank -> top 5 to LLM.

**Eval hit**:

```
Faithfulness      1.000
Answer Relevancy  0.386
Context Precision 0.333
```

Faithfulness at 1.0 -- no hallucination. Everything came from what was retrieved. 0.386 is the one I couldn't ignore. I asked "what is the use of Kubernetes?" Retrieval pulled an Overview chunk covering orchestration, scaling, networking, deployments -- all of it, one block. The model summarized that block faithfully. Didn't matter that it wasn't what I asked.

The reranker scored that chunk **-0.44**. Negative. It knew something was wrong. Still ended up in the final 5 because nothing better existed upstream.

**Problem**: Chunks at 2,813 average characters. The two sentences that actually answer the question are somewhere inside a block about twelve other things. Dense retrieval can't isolate them. The reranker can't fix what the chunker broke.

### V2 -- Re-chunk + positive filter

<video src="https://github.com/vishal24p/learning/raw/main/rag/assets/version2.gif" controls width="100%"></video>

**Fix**: Buffer 5, breakpoint percentile 80. Same corpus, same embed model. Re-indexed into a fresh database. 14,734 chunks. Average **1,045 characters**.

**Added `filter_positive_chunks()`** -- anything below reranker score zero drops out before the LLM sees it. If everything is negative, keep the single best chunk so the model still has something to ground on.

```
Before                         After
Faithfulness      1.000       Faithfulness      0.889
Answer Relevancy  0.386       Answer Relevancy  0.661
Context Precision 0.333       Context Precision 1.000
```

Context precision went from 0.333 to 1.000. Every chunk in the context window was relevant. Answer relevancy nearly doubled. Faithfulness dropped from 1.000 to 0.889 -- the model started saying things beyond the retrieved context when the chunks were tighter. Worth the trade-off for answers that actually answer the question.

### V3 -- Pre-pipeline LLM safety guard

A separate small model call (`llama-guard3:1b`) runs *before* any retrieval. The model classifies `safe` or `unsafe`. On `unsafe`, the pipeline stops: no embedding call, no Postgres query, no generator call, no RAGAS button. The refusal is logged with the Llama Guard category code (S1..S13) for audit.

The guard is gated by `GUARD_ENABLED=true` in `.env`. Flip it to `false` and the pipeline runs unguarded -- one env var to compare against baseline.

## Run it

```bash
ollama pull nomic-embed-text-v2-moe
ollama pull llama-guard3:1b
ollama pull minimax-m3:cloud

# Database + index
python scripts/setup_paradedb.py
python scripts/apply_migration.py

# Index the corpus
python scripts/reindex_v2.py

# UI
run_ui.cmd
```
