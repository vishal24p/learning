"""Unit tests for HybridRetriever (RRF fusion)."""
from __future__ import annotations


class FakeDense:
    def __init__(self, hits):
        self._hits = hits
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        return self._hits[:top_k]


class FakeSparse:
    def __init__(self, hits):
        self._hits = hits
        self.calls = []

    def retrieve(self, query, top_k):
        self.calls.append((query, top_k))
        return self._hits[:top_k]


def _hit(chunk_id, content, score):
    return {"chunk_id": chunk_id, "content": content, "score": score}


def test_hybrid_fuses_with_rrf_and_returns_top_k():
    dense_hits = [
        _hit(1, "alpha", 0.99),
        _hit(2, "beta", 0.90),
        _hit(3, "gamma", 0.80),
    ]
    sparse_hits = [
        _hit(2, "beta", 1.10),  # rank 1
        _hit(1, "alpha", 0.95),  # rank 2
        _hit(4, "delta", 0.50),  # rank 3
    ]

    from src.retrieval.hybrid_retriever import HybridRetriever

    hybrid = HybridRetriever(
        "postgresql://x",
        dense_retriever=FakeDense(dense_hits),
        sparse_retriever=FakeSparse(sparse_hits),
        rrf_k=60,
    )

    results = hybrid.retrieve("q", top_k_dense=3, top_k_sparse=3, top_k=3)

    # id=2 is in both lists (dense rank1, sparse rank?) — recompute precisely.
    # dense ranks (1..3) = ids 1,2,3
    # sparse ranks (1..3) = ids 2,1,4
    score_2 = 1 / (2 + 60) + 1 / (1 + 60)  # dense rank2 + sparse rank1
    score_1 = 1 / (1 + 60) + 1 / (2 + 60)  # dense rank1 + sparse rank2
    score_3 = 1 / (3 + 60) + 0             # sparse-only
    score_4 = 0 + 1 / (3 + 60)             # dense has nothing for id=4

    by_id = {r["chunk_id"]: r["score"] for r in results}
    assert set(by_id) >= {1, 2, 3}
    for chunk_id, expected in [(2, score_2), (1, score_1), (3, score_3), (4, score_4)]:
        if chunk_id in by_id:
            assert abs(by_id[chunk_id] - expected) < 1e-9
    assert len(results) == 3


def test_hybrid_calls_each_retriever_with_correct_top_k():
    dense = FakeDense([])
    sparse = FakeSparse([])
    from src.retrieval.hybrid_retriever import HybridRetriever

    HybridRetriever(
        "postgresql://x",
        dense_retriever=dense,
        sparse_retriever=sparse,
    ).retrieve("query", top_k_dense=10, top_k_sparse=12, top_k=5)

    assert dense.calls == [("query", 10)]
    assert sparse.calls == [("query", 12)]


def test_hybrid_handles_zero_overlap():
    dense_hits = [_hit(1, "alpha", 0.99)]
    sparse_hits = [_hit(2, "beta", 1.10)]

    from src.retrieval.hybrid_retriever import HybridRetriever

    results = HybridRetriever(
        "postgresql://x",
        dense_retriever=FakeDense(dense_hits),
        sparse_retriever=FakeSparse(sparse_hits),
    ).retrieve("q", top_k_dense=1, top_k_sparse=1, top_k=2)

    # Both rank-1 => equal contribution => tie; order is not guaranteed
    assert {r["chunk_id"] for r in results} == {1, 2}
    expected = 1 / (1 + 60)
    for r in results:
        assert abs(r["score"] - expected) < 1e-9
