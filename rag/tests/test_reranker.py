"""Unit tests for CrossEncoderReranker + RerankedRetriever.

Fake scorer injection avoids downloading the model in CI.
"""
from __future__ import annotations


def _hit(chunk_id, content, score):
    return {"chunk_id": chunk_id, "content": content, "score": score}


def test_reranker_uses_score_fn_and_returns_top_n():
    from src.retrieval.reranker import CrossEncoderReranker

    # The fake scorer rewards chunks whose content 'heading' word count is higher.
    def fake_score(query, text):
        return float(text.count("h"))

    candidates = [
        _hit(1, "h h h h h h", 0.001),
        _hit(2, "no relevance here", 0.001),
        _hit(3, "h h h", 0.001),
        _hit(4, "h", 0.001),
    ]
    r = CrossEncoderReranker(score_fn=fake_score)
    out = r.rerank("find heading", candidates, top_n=2)

    assert len(out) == 2
    assert out[0]["chunk_id"] == 1   # six h's -> top
    assert out[1]["chunk_id"] == 3   # three h's -> second
    # original RRF scores are replaced with reranker scores
    for o in out:
        assert "score" in o
        assert o["score"] != 0.001


def test_reranker_returns_empty_for_empty_input():
    from src.retrieval.reranker import CrossEncoderReranker

    r = CrossEncoderReranker(score_fn=lambda q, t: 0.0)
    assert r.rerank("q", []) == []


def test_reranked_retriever_calls_hybrid_with_pool_size():
    from src.retrieval.reranked_retriever import RerankedRetriever

    class FakeHybrid:
        def __init__(self):
            self.calls = None

        def retrieve(self, query, top_k_dense, top_k_sparse, top_k):
            self.calls = (query, top_k_dense, top_k_sparse, top_k)
            return [
                _hit(1, "first", 0.9),
                _hit(2, "second", 0.5),
            ]

    class FakeReranker:
        def rerank(self, query, candidates, top_n):
            return [c for c in candidates][:top_n]

    hybrid = FakeHybrid()
    rr = RerankedRetriever(
        "postgresql://x",
        hybrid=hybrid,
        reranker=FakeReranker(),
    )

    rrf, reranked = rr.retrieve("query", candidates=20, top_n=5)
    assert hybrid.calls == ("query", 10, 10, 20)
    assert len(reranked) == 2  # only 2 candidates existed; nothing invented
