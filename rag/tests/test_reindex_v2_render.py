"""Verify the reindex_v2 migration renderer always emits exactly ONE USING bm25 index.

Why: a bad rename earlier produced two ``chunks_v2_bm25`` + ``chunks_bm25``
indexes on the same relation, breaking the migration with
"a relation may only have one USING bm25 index".
"""
from __future__ import annotations

import re
from pathlib import Path


def _read_source() -> str:
    return (Path(__file__).parent.parent / "src" / "db" / "migrations" / "001_init.sql").read_text(
        encoding="utf-8"
    )


def _render(table: str) -> str:
    sql = _read_source()
    sql = sql.replace("chunks_bm25", f"{table}_bm25")
    sql = sql.replace("chunks_embedding_hnsw", f"{table}_embedding_hnsw")
    sql = re.sub(r"\bchunks\b", table, sql)
    sql = sql.replace("embedding    vector,", "embedding    vector(768),")
    return sql


def test_render_emits_exactly_one_bm25_index():
    rendered = _render("chunks_v2")
    bm25_count = len(re.findall(r"USING\s+bm25\b", rendered))
    hnsw_count = len(re.findall(r"USING\s+hnsw\b", rendered))
    assert bm25_count == 1, f"expected exactly one USING bm25, got {bm25_count}\n{rendered}"
    assert hnsw_count == 1, f"expected exactly one USING hnsw, got {hnsw_count}\n{rendered}"


def test_render_index_name_uses_table_prefix():
    rendered = _render("chunks_v2")
    assert "chunks_v2_bm25" in rendered
    assert "chunks_v2_embedding_hnsw" in rendered
    assert "chunks_bm25" not in rendered
    assert "chunks_embedding_hnsw" not in rendered
    # The plain `chunks` token (table name) must be rewritten too.
    assert "ON chunks_v2" in rendered
    assert "ON chunks\n" not in rendered


def test_render_unknown_table_name_rejected_by_store():
    """PostgresStore rejects table names that fail the safe-identifier check."""
    from src.storage.postgres_store import PostgresStore

    try:
        PostgresStore("irrelevant", table="chunks_v2; DROP TABLE foo")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unsafe table name")


def test_store_default_table_is_chunks():
    from src.storage.postgres_store import PostgresStore

    s = PostgresStore.__new__(PostgresStore)
    # Without running __init__, the default table name should match.
    # We don't call __init__ because that would attempt network calls.
    import inspect

    sig = inspect.signature(PostgresStore.__init__)
    assert "table" in sig.parameters
    assert sig.parameters["table"].default == "chunks"
    s.table = "chunks"
