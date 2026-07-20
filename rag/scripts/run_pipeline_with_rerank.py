r"""End-to-end demo: same query, three views side by side.

   1. RRF only (existing behavior)
   2. RRF + cross-encoder rerank (new behavior)
   3. (Optional) the reranked top_n fed into Generator -> answer

Usage (PowerShell, venv active):
    $env:PYTHONPATH = "C:\Users\visha\rag"
    python scripts\run_pipeline_with_rerank.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.guard.pipeline import guard_or_terminate  # noqa: E402
from src.guard.reason import PipelineTerminated  # noqa: E402
from src.query_rewrite import rewrite_query  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.reranked_retriever import RerankedRetriever  # noqa: E402

QUERY = "How do H1, H2, H3 heading levels work in markdown?"


def show(name, hits, n_chars=160):
    print(f"\n== {name} ({len(hits)} hits) ==")
    for i, h in enumerate(hits, 1):
        preview = h["content"].replace("\n", " ")[:n_chars]
        print(f"  {i}. id={h['chunk_id']:<5}  score={h['score']:.4f}  | {preview}...")


def main():
    original_query = QUERY
    print(
        f"[cfg] rerank_model={settings.rerank_model} "
        f"candidates={settings.rerank_candidates} top_n={settings.rerank_top_n} "
        f"guard={'on' if settings.guard_enabled else 'off'} "
        f"guard_model={settings.guard_model} "
        f"query_rewrite={'on' if settings.query_rewrite_enabled else 'off'} "
        f"query_rewrite_model={settings.query_rewrite_model}"
    )

    try:
        query = guard_or_terminate(original_query)
    except PipelineTerminated as exc:
        print("\n== GUARD: TERMINATE ==")
        print(f"reason : {exc.decision.reason}")
        print(f"refusal: {exc.decision.refusal}")
        return 0

    # Single-query rewrite for retrieval. Runs after guard => safe,
    # before hybrid/rerank.
    query = rewrite_query(query)
    if query != original_query:
        print(f"\n[rewrite] {original_query}\n     -> {query}")

    hybrid = HybridRetriever(settings.db_url)
    rrf_pool = hybrid.retrieve(query, top_k_dense=10, top_k_sparse=10, top_k=settings.rerank_candidates)
    show("RRF pool", rrf_pool)

    rr = RerankedRetriever(settings.db_url)
    # Direct call uses Hybrid + Rerank with one embed/DB roundtrip per side.
    rrf_again, reranked = rr.retrieve(
        query,
        top_k_dense=10,
        top_k_sparse=10,
        candidates=settings.rerank_candidates,
        top_n=settings.rerank_top_n,
    )

    show("RRF (top 5 for comparison)", rrf_again[: settings.rerank_top_n])
    show("RERANKED (final top 5)", reranked)

    overlap_ids = {h["chunk_id"] for h in rrf_again[: settings.rerank_top_n]}.intersection(
        {h["chunk_id"] for h in reranked}
    )
    print(f"\n[summary] rerank kept/added {len(overlap_ids)} of RRF top-5; "
          f"swapped in {settings.rerank_top_n - len(overlap_ids)} from the wider pool")
    return 0


if __name__ == "__main__":
    sys.exit(main())
