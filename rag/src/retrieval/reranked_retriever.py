"""Composite retriever: Hybrid (RRF) -> CrossEncoderReranker.

Pipeline:
   query
     -> HybridRetriever.retrieve(query, top_k_dense=N, top_k_sparse=N, top_k=pool)
        (returns a wide list of fused candidates)
     -> CrossEncoderReranker.rerank(query, candidates, top_n=final)
        (re-scores by query/chunk relevance and keeps the best)
"""
from __future__ import annotations

import logging
import time

from src.config import settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

log = logging.getLogger(__name__)


class RerankedRetriever:
    def __init__(
        self,
        connection_string: str,
        hybrid: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.hybrid = hybrid or HybridRetriever(connection_string)
        self.reranker = reranker or CrossEncoderReranker()

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        candidates: int | None = None,
        top_n: int | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Return (rrf_candidates, reranked_top_n).

        The first list lets callers inspect what the hybrid produced before
        the reranker acted on it. The second list is the actual answer.
        """
        pool = candidates if candidates is not None else settings.rerank_candidates
        final = top_n if top_n is not None else settings.rerank_top_n

        log.info("pipeline start: hybrid pool=%d rerank top_n=%d", pool, final)
        t0 = time.perf_counter()
        rrf_pool = self.hybrid.retrieve(
            query,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            top_k=pool,
        )
        reranked = self.reranker.rerank(query, rrf_pool, top_n=final)
        log.info(
            "pipeline end: rrf=%d -> reranked=%d (total %.1f ms)",
            len(rrf_pool), len(reranked),
            (time.perf_counter() - t0) * 1000.0,
        )
        return rrf_pool, reranked
