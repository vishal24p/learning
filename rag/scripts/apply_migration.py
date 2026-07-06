r"""Apply migrations/001_init.sql using psycopg (no psql needed).

The HNSW index needs the embedding column dim to be known. We probe the dim
once by calling the configured Ollama embed model on a short string before
running the migration.

Usage (Windows PowerShell, venv active):
    $env:PYTHONPATH = "C:\Users\visha\rag"
    python scripts\apply_migration.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402


def probe_embedding_dim() -> int | None:
    """Return the embedding dim of the configured model, or None on failure."""
    try:
        from src.embeddings.embedder import Embedder

        emb = Embedder()
        vec = emb.embed_chunk("dimension probe")
        return len(vec)
    except Exception as exc:  # noqa: BLE001
        print(f"[probe] could not reach Ollama at {settings.ollama_base_url}: {exc}")
        return None


def render_migration(dim: int | None) -> str:
    sql_path = Path(__file__).parent.parent / "src" / "db" / "migrations" / "001_init.sql"
    sql = sql_path.read_text(encoding="utf-8")

    if dim is not None:
        sql = sql.replace("embedding    vector,", f"embedding    vector({dim}),   -- dim probed from Ollama")
    else:
        sql = re.sub(
            r"CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw.*?;\n",
            "-- HNSW index skipped (no dim available; dense search still works brute-force)\n",
            sql,
            flags=re.DOTALL,
        )
    return sql


def main() -> int:
    print("[probe] measuring embedding dimension from Ollama ...")
    dim = probe_embedding_dim()
    if dim is None:
        print("[probe] proceeding without dim; HNSW index will be skipped")
    else:
        print(f"[probe] embedding dim = {dim}")

    sql = render_migration(dim)
    print("[migrate] applying SQL:")
    for line in sql.splitlines():
        if line.strip():
            print(f"   {line}")

    import psycopg

    # Drop any stale BM25 index from earlier (pre-fix) migrations.
    # CREATE INDEX IF NOT EXISTS would silently skip the recreation.
    with psycopg.connect(settings.db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS chunks_bm25;")

    with psycopg.connect(settings.db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)

    print("[migrate] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
