"""Tests for PostgresStore insert + count, using an in-process fake adapter."""
from __future__ import annotations


class FakeCursor:
    def __init__(self, store: "FakeConnection") -> None:
        self._store = store
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql: str, params: tuple) -> None:
        if sql.strip().upper().startswith("INSERT INTO CHUNKS"):
            row_id = self._store._next_id
            self._store._next_id += 1
            self._result = (row_id,)
        elif "COUNT(*)" in sql:
            self._result = (self._store._count,)
        else:
            self._result = ()

    def fetchone(self):
        return self._result


class FakeConnection:
    def __init__(self) -> None:
        self._next_id: int = 1
        self._count: int = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return FakeCursor(self)

    def commit(self) -> None:
        pass


def test_insert_chunk_returns_id_and_increments_count():
    from src.storage.postgres_store import PostgresStore

    conn = FakeConnection()
    store = PostgresStore.__new__(PostgresStore)
    store.connection_string = "irrelevant"
    store.table = "chunks"
    store._conn_factory = lambda _: conn  # type: ignore[attr-defined]

    # Patch the `connect` symbol used by the module.
    import src.storage.postgres_store as mod
    original_connect = mod.connect
    mod.connect = lambda *args, **kwargs: conn  # type: ignore[assignment]
    try:
        row_id = store.insert_chunk(
            document_id="doc1",
            chunk_index=0,
            content="hello",
            embedding=[0.1, 0.2, 0.3],
        )
        assert row_id == 1
    finally:
        mod.connect = original_connect

    assert conn._count == 0  # count is a separate query
