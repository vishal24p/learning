"""ParadeDB BM25-based sparse retriever."""
from __future__ import annotations

import logging
import re
import time

from psycopg import connect

from src.config import settings

log = logging.getLogger(__name__)

_SAFE_TABLE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class SparseRetriever:
    def __init__(
        self,
        connection_string: str,
        table: str | None = None,
    ) -> None:
        self.connection_string = connection_string
        self.table = table or settings.chunks_table
        if not _SAFE_TABLE.match(self.table):
            raise ValueError(f"Invalid chunks table name: {self.table!r}")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        sql = (
            f"SELECT id, content, paradedb.score(id) AS score "
            f"FROM {self.table} WHERE content @@@ %s ORDER BY score DESC LIMIT %s"
        )
        log.debug("sql: %s params=(query[len=%d], top_k=%d)", sql, len(query), top_k)

        t = time.perf_counter()
        with connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (query, top_k))
                rows = cur.fetchall()
        log.debug("bm25 scan returned %d rows (%.1f ms)", len(rows),
                  (time.perf_counter() - t) * 1000.0)

        return [
            {"chunk_id": row[0], "content": row[1], "score": float(row[2])}
            for row in rows
        ]
