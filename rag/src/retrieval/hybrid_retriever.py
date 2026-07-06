"""Reciprocal Rank Fusion (RRF) over dense + sparse retrievers.

Each retriever returns a list of {chunk_id, content, score}. We fuse by rank,
not by raw score, so dense cosine and sparse BM25 can be combined cleanly.

score(c) = 1 / (rank_dense(c) + K) + 1 / (rank_sparse(c) + K)
"""
from __future__ import annotations

import logging
import time

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever

log = logging.getLogger(__name__)

DEFAULT_K = 60  # standard RRF constant


def _timed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


class HybridRetriever:
    def __init__(
        self,
        connection_string: str,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        rrf_k: int = DEFAULT_K,
    ) -> None:
        self.dense = dense_retriever or DenseRetriever(connection_string)
        self.sparse = sparse_retriever or SparseRetriever(connection_string)
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 10,
        top_k_sparse: int = 10,
        top_k: int = 5,
    ) -> list[dict]:
        log.info(
            "retrieval start dense_top=%d sparse_top=%d rrf_top=%d",
            top_k_dense, top_k_sparse, top_k,
        )
        t = time.perf_counter()
        dense_hits = self.dense.retrieve(query, top_k=top_k_dense)
        log.info("dense hits=%d (%.1f ms)", len(dense_hits), _timed_ms(t))
        t = time.perf_counter()
        sparse_hits = self.sparse.retrieve(query, top_k=top_k_sparse)
        log.info("sparse hits=%d (%.1f ms)", len(sparse_hits), _timed_ms(t))

        fused = self._fuse(dense_hits, sparse_hits, top_k)
        log.info("rrf fused=%d", len(fused))
        return fused

    def _fuse(
        self,
        dense_hits: list[dict],
        sparse_hits: list[dict],
        top_k: int,
    ) -> list[dict]:
        # Preserve content from whichever side surfaced a chunk first;
        # both retrievers return the same content for the same chunk_id.
        content_by_id: dict[int, str] = {}
        for hit in dense_hits:
            content_by_id.setdefault(hit["chunk_id"], hit["content"])
        for hit in sparse_hits:
            content_by_id.setdefault(hit["chunk_id"], hit["content"])

        fused: dict[int, float] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            fused[hit["chunk_id"]] = fused.get(hit["chunk_id"], 0.0) + 1.0 / (rank + self.rrf_k)
        for rank, hit in enumerate(sparse_hits, start=1):
            fused[hit["chunk_id"]] = fused.get(hit["chunk_id"], 0.0) + 1.0 / (rank + self.rrf_k)

        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return [
            {
                "chunk_id": chunk_id,
                "content": content_by_id[chunk_id],
                "score": float(score),
            }
            for chunk_id, score in ordered[:top_k]
        ]
