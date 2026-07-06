"""Tests for PostgresStore.insert_chunks_many -- the batched multi-row INSERT."""
from __future__ import annotations


class FakeCursor:
    def __init__(self) -> None:
        self._fetch = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql: str, params) -> None:
        # `params` is a flat list of 4-tuples * row_count; identify row count
        # by parameter-set count = len(ps)/4.
        row_count = len(params) // 4
        next_id = 1000
        self._fetch = [(next_id + i,) for i in range(row_count)]

    def fetchall(self):
        return self._fetch


class FakeConn:
    def __init__(self) -> None:
        self.executed_sql: list[str] = []
        self.cursor_obj = FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self.cursor_obj

    def commit(self) -> None:
        pass


def test_insert_chunks_many_single_batch():
    from src.storage.postgres_store import PostgresStore
    import src.storage.postgres_store as mod

    conn = FakeConn()

    store = PostgresStore.__new__(PostgresStore)
    store.connection_string = "irrelevant"
    store.table = "chunks"

    original_connect = mod.connect
    mod.connect = lambda *a, **k: conn
    try:
        ids = store.insert_chunks_many([
            {"document_id": "d1", "chunk_index": 0, "content": "a", "embedding": [0.1]},
            {"document_id": "d1", "chunk_index": 1, "content": "b", "embedding": [0.2]},
            {"document_id": "d1", "chunk_index": 2, "content": "c", "embedding": [0.3]},
        ])
    finally:
        mod.connect = original_connect

    assert ids == [1000, 1001, 1002], ids


def test_insert_chunks_many_multiple_batches():
    """A 600-row list with batch_size=200 should run 3 statements, all in one
    PostgreSQL connection."""
    from src.storage.postgres_store import PostgresStore
    import src.storage.postgres_store as mod

    store = PostgresStore.__new__(PostgresStore)
    store.connection_string = "x"
    store.table = "chunks"

    cursor_calls = {"n": 0}

    class _TrackingCursor(FakeCursor):
        def execute(self, sql, params):
            cursor_calls["n"] += 1
            super().execute(sql, params)

    class _TrackingConn(FakeConn):
        def __init__(self):
            super().__init__()
            self.cursor_obj = _TrackingCursor()

    conn = _TrackingConn()
    original_connect = mod.connect
    mod.connect = lambda *a, **k: conn
    try:
        items = [
            {"document_id": "d", "chunk_index": i, "content": f"c{i}", "embedding": [0.0]}
            for i in range(600)
        ]
        ids = store.insert_chunks_many(items, batch_size=200)
    finally:
        mod.connect = original_connect

    assert cursor_calls["n"] == 3, f"expected 3 batches, got {cursor_calls['n']}"
    assert len(ids) == 600


def test_insert_chunks_many_empty():
    from src.storage.postgres_store import PostgresStore
    store = PostgresStore.__new__(PostgresStore)
    store.connection_string = "x"
    store.table = "chunks"
    assert store.insert_chunks_many([]) == []
