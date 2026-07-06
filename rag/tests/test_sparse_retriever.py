"""Unit test: SparseRetriever shapes results uniformly and uses @@@ + score."""
from __future__ import annotations


class FakeCursor:
    def __init__(self, rows: list[tuple]) -> None:
        self._rows = rows
        self.executed: tuple[str, tuple] | None = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql: str, params: tuple):
        self.executed = (sql, params)

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, rows):
        self.cursor_obj = FakeCursor(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self.cursor_obj


def test_sparse_retriever_returns_uniform_shape_and_uses_pgsearch():
    rows = [
        (1, "heading levels explained", 1.42),
        (2, "how to write h2", 0.91),
    ]

    captured: dict = {}

    def fake_connect(*args, **kwargs):
        rows_ref = rows

        class _C(FakeConn):
            def __init__(self_inner) -> None:  # noqa: N805
                super().__init__(rows_ref)
                captured["conn"] = self_inner
        return _C()

    import src.retrieval.sparse_retriever as mod
    original = mod.connect
    mod.connect = fake_connect
    try:
        retriever = mod.SparseRetriever("postgresql://x")
        results = retriever.retrieve("heading levels", top_k=2)
    finally:
        mod.connect = original

    assert results == [
        {"chunk_id": 1, "content": "heading levels explained", "score": 1.42},
        {"chunk_id": 2, "content": "how to write h2", "score": 0.91},
    ]

    sql, params = captured["conn"].cursor_obj.executed
    assert "@@@" in sql
    assert "paradedb.score(id)" in sql
    assert params == ("heading levels", 2)
