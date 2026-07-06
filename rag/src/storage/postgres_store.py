"""Thin Postgres wrapper around a chunk table.

The ParadeDB BM25 index (see migrations/001_init.sql) indexes ``content``
automatically; we only need to write the row.

The constructor accepts an optional ``table`` so the same store can be
pointed at ``chunks`` (default) or a versioned variant like ``chunks_v2``.
"""
from __future__ import annotations

import re

from psycopg import connect
from psycopg.types.json import Jsonb


_SAFE_TABLE_NAME = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class PostgresStore:
    def __init__(self, connection_string: str, table: str = "chunks") -> None:
        if not _SAFE_TABLE_NAME.match(table):
            raise ValueError(
                f"Invalid table name {table!r}; only [a-zA-Z_][a-zA-Z0-9_]* allowed"
            )
        self.connection_string = connection_string
        self.table = table

    def _insert_one_cursor(self, cur, document_id, chunk_index, content, embedding):
        cur.execute(
            f"""
            INSERT INTO {self.table} (
                document_id,
                chunk_index,
                content,
                embedding
            )
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (document_id, chunk_index, content, embedding),
        )
        return cur.fetchone()[0]

    def insert_chunk(
        self,
        document_id: str | None,
        chunk_index: int,
        content: str,
        embedding: list[float],
    ) -> int:
        """Insert one chunk and return its row id."""
        with connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                row_id = self._insert_one_cursor(
                    cur, document_id, chunk_index, content, embedding
                )
            conn.commit()
        return row_id

    def insert_chunks_many(self, items: list[dict], *, batch_size: int = 200) -> list[int]:
        """Insert many chunks in a single round-trip per batch.

        Each item is a dict with keys: ``document_id``, ``chunk_index``,
        ``content``, ``embedding``. Returns the list of generated row ids
        in the same order.

        Why not use ``psycopg.extras.execute_values``?
        The ``psycopg`` 3.x stdlib doesn't ship ``psycopg.extras`` as a
        top-level module. Building the multi-row INSERT manually with
        parameterized values is portable, safe, and uses a single
        server round-trip per batch.
        """
        if not items:
            return []

        rows: list[int] = []
        placeholder_block_one = "(%s, %s, %s, %s)"
        with connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                start = 0
                n = len(items)
                while start < n:
                    end = min(start + batch_size, n)
                    chunk = items[start:end]

                    placeholders = ",".join([placeholder_block_one] * len(chunk))
                    args: list = []
                    for it in chunk:
                        args.extend(
                            [
                                it.get("document_id"),
                                it["chunk_index"],
                                it["content"],
                                it["embedding"],
                            ]
                        )

                    sql = (
                        f"INSERT INTO {self.table} "
                        f"(document_id, chunk_index, content, embedding) "
                        f"VALUES {placeholders} RETURNING id"
                    )
                    cur.execute(sql, args)
                    rows.extend(r[0] for r in cur.fetchall())
                    start = end
            conn.commit()
        return rows

    def count(self) -> int:
        with connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table}")
                return cur.fetchone()[0]
