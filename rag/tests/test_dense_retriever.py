"""Unit test: DenseRetriever uses <=> and uniform result shape."""
from __future__ import annotations


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self.executed = (sql, params)

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, rows):
        self._cursor = FakeCursor(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self._cursor


class FakeEmbedder:
    def __init__(self, vec):
        self._vec = vec

    def embed_chunk(self, text):
        return self._vec


def test_dense_retriever_uniform_shape_and_cosine():
    rows = [(7, "adaptive content", 0.81)]

    def fake_connect(*a, **k):
        return FakeConn(rows)

    import src.retrieval.dense_retriever as mod
    original = mod.connect
    mod.connect = fake_connect
    try:
        retriever = mod.DenseRetriever("postgresql://x", embedder=FakeEmbedder([0.1, 0.2]))
        out = retriever.retrieve("adaptive content", top_k=1)
    finally:
        mod.connect = original

    assert out == [{"chunk_id": 7, "content": "adaptive content", "score": 0.81}]
    sql, params = retriever.retrieve.__wrapped__.__name__ if False else None, None  # noop

    # Just re-execute to capture sql/params since we already consumed once:
    cap = {}

    def cap_connect(*a, **k):
        c = FakeConn([])
        cap["c"] = c
        return c

    mod.connect = cap_connect
    try:
        retriever2 = mod.DenseRetriever("postgresql://x", embedder=FakeEmbedder([0.1]))
        retriever2.retrieve("q", top_k=3)
    finally:
        mod.connect = original

    sql, params = cap["c"]._cursor.executed
    assert "<=>" in sql
    assert params == ([0.1], [0.1], 3)
