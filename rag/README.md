# RAG over Kubernetes Docs

Hybrid (BM25 + pgvector) semantic RAG with a pre-pipeline LLM safety guard. Fully local: Postgres/ParadeDB + Ollama + sentence-transformers reranker + Streamlit. No API keys.

## Stack

| Layer | Tool |
|---|---|
| Database + Index | ParadeDB (pgvector + BM25) |
| Embeddings | `nomic-embed-text-v2-moe` via Ollama |
| Semantic Chunker | LlamaIndex `SemanticSplitterNodeParser` |
| Hybrid Fusion | RRF (Reciprocal Rank Fusion, k=60) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generator | `minimax-m3:cloud` via Ollama |
| Eval | RAGAS (faithfulness, answer relevancy, context precision) |
| UI | Streamlit |

## Architecture

**V1**

![V1 architecture](https://github.com/vishal24p/learning/raw/main/rag/assets/version1-arch.webp)

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

##
### V1.1 -- Re-chunk + positive filter

<img src="https://github.com/vishal24p/learning/raw/main/rag/assets/version2.gif" alt="V2 demo" width="100%">

> [version2.gif](https://github.com/vishal24p/learning/blob/main/rag/assets/version2.gif) for the animated loop. (Filename is V2 on disk; this is V1.1 in the pipeline.)

Chunker retuned to buffer 5, breakpoint percentile 80. Same corpus, same embed model, fresh database. 14,734 chunks. Average **1,045 chars**.

`filter_positive_chunks()` added post-rerank: anything below reranker score zero drops out before the LLM sees it. If every candidate is negative, keep the single best so the model still has grounding.

```
V1                              V1.1
Faithfulness      1.000        Faithfulness      0.889
Answer Relevancy  0.386        Answer Relevancy  0.661
Context Precision 0.333        Context Precision 1.000
```

Context precision 0.333 → 1.000. Every chunk in the context window is now relevant. Answer relevancy nearly doubles. Faithfulness drops from 1.000 → 0.889 because the model starts going beyond retrieved context when chunks are tight enough to answer from. Worth the trade-off; answers now answer the question.

