"""Dense retriever via Ollama embeddings + pgvector cosine distance."""
from __future__ import annotations

import logging
import re
import time

from psycopg import connect

from src.config import settings
from src.embeddings.embedder import Embedder

log = logging.getLogger(__name__)

_SAFE_TABLE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class DenseRetriever:
    def __init__(
        self,
        connection_string: str,
        embedder: Embedder | None = None,
        table: str | None = None,
    ) -> None:
        self.connection_string = connection_string
        self.embedder = embedder or Embedder()
        self.table = table or settings.chunks_table
        if not _SAFE_TABLE.match(self.table):
            raise ValueError(f"Invalid chunks table name: {self.table!r}")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        log.debug("embed query (dim probe): start")
        t = time.perf_counter()
        query_embedding = self.embedder.embed_chunk(query)
        embed_ms = (time.perf_counter() - t) * 1000.0
        log.info("embed query dim=%d (%.1f ms)", len(query_embedding), embed_ms)

        sql = (
            f"SELECT id, content, 1 - (embedding <=> %s::vector) AS score "
            f"FROM {self.table} ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        log.debug("sql: %s params=(vec[%d], vec[%d], top_k=%d)", sql,
                  len(query_embedding), len(query_embedding), top_k)

        t = time.perf_counter()
        with connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (query_embedding, query_embedding, top_k))
                rows = cur.fetchall()
        log.debug("pgvector scan returned %d rows", len(rows))

        return [
            {"chunk_id": row[0], "content": row[1], "score": float(row[2])}
            for row in rows
        ]
