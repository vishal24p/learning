"""Unit tests for CrossEncoderReranker + RerankedRetriever.

Fake scorer injection avoids downloading the model in CI.
"""
from __future__ import annotations

import pytest


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


def test_reranker_keeps_negative_scores_in_ranked_top_n():
    from src.retrieval.reranker import CrossEncoderReranker

    scores = {"best": -0.5, "next": -1.0, "last": -2.0}
    r = CrossEncoderReranker(score_fn=lambda q, text: scores[text])

    out = r.rerank("q", [_hit(i, text, 0.0) for i, text in enumerate(scores)], top_n=3)

    assert [item["content"] for item in out] == ["best", "next", "last"]
    assert [item["score"] for item in out] == [-0.5, -1.0, -2.0]


def test_reranker_softmax_normalizes_scores_and_preserves_order(monkeypatch):
    from dataclasses import replace
    import src.retrieval.reranker as reranker_module
    from src.retrieval.reranker import CrossEncoderReranker

    monkeypatch.setattr(
        reranker_module,
        "settings",
        replace(reranker_module.settings, rerank_use_softmax=True),
    )
    raw_scores = {"high": 1000.0, "middle": 999.0, "low": 998.0}
    r = CrossEncoderReranker(score_fn=lambda q, text: raw_scores[text])

    out = r.rerank("q", [_hit(i, text, 0.0) for i, text in enumerate(raw_scores)], top_n=3)

    assert [item["content"] for item in out] == ["high", "middle", "low"]
    assert sum(item["score"] for item in out) == pytest.approx(1.0)
    assert out[0]["score"] > out[1]["score"] > out[2]["score"]


def test_reranker_softmax_disabled_returns_raw_scores(monkeypatch):
    from dataclasses import replace
    import src.retrieval.reranker as reranker_module
    from src.retrieval.reranker import CrossEncoderReranker

    monkeypatch.setattr(
        reranker_module,
        "settings",
        replace(reranker_module.settings, rerank_use_softmax=False),
    )
    raw_scores = {"high": 2.0, "low": -1.0}
    r = CrossEncoderReranker(score_fn=lambda q, text: raw_scores[text])

    out = r.rerank("q", [_hit(i, text, 0.0) for i, text in enumerate(raw_scores)], top_n=2)

    assert [item["score"] for item in out] == [2.0, -1.0]


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
