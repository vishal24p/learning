"""Reindex the markdown corpus into a *new database* using the existing
indexer (``src.ingestion.index.index_docs``), tuned chunker params, and a
versioned table (``chunks_v2``).

The original database (default ``rag``) and its ``chunks`` table stay
untouched. To switch the live pipeline over, flip ``settings.db_url`` in
the environment / ``.env`` (e.g. ``DB_NAME=rag_v2``).

Inputs
------
   --corpus PATH       root directory (default: K8s website content/en)
   --include-subdir  repeatable subdirectory under --corpus to walk
                       (default: docs)
   --db    NAME        target database name (default: rag_v2)
   --table  NAME       chunk table inside that db (default: chunks_v2)
   --buffer-size N     SemanticSplitterNodeParser buffer_size (default 5)
   --threshold P       breakpoint percentile threshold (default 80)

   By default the walker only descends into ``website-main/content/en/docs``
   (concepts/tasks/tutorials/reference/setup/...); everything else (blog,
   careers, includes, images, etc.) is skipped both via ``--include-subdir``
   and via the curated exclusion list baked into ``load_markdown_files``.

Run by hand:
    python scripts\\reindex_v2.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="reindex markdown into rag_v2 / chunks_v2",
    )
    ap.add_argument(
        "--corpus",
        default=r"C:\Users\visha\rag\website-main\content\en",
        help="root directory containing *.md files",
    )
    ap.add_argument(
        "--include-subdir",
        action="append",
        default=["docs"],
        help="subdirectory under --corpus to walk (repeatable; default: docs)",
    )
    ap.add_argument("--db", default="rag_v2", help="target database name")
    ap.add_argument(
        "--table",
        default="chunks_v2",
        help="chunk table name inside the target database (alphanumeric/_ only)",
    )
    ap.add_argument(
        "--buffer-size",
        type=int,
        default=5,
        help="SemanticSplitterNodeParser buffer_size (default 5)",
    )
    ap.add_argument(
        "--threshold",
        type=int,
        default=80,
        help="breakpoint_percentile_threshold (default 80)",
    )
    return ap.parse_args()


def db_url_for(target_db: str) -> str:
    """Return settings.db_url with the database portion swapped."""
    return settings.db_url.rsplit("/", 1)[0] + "/" + target_db


def ensure_database_exists(target_db: str) -> None:
    """Create the target database if it doesn't already exist."""
    import psycopg

    admin_url = db_url_for("postgres")
    with psycopg.connect(admin_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (target_db,))
            if cur.fetchone() is None:
                print(f"[setup] creating database {target_db}")
                cur.execute(f'CREATE DATABASE "{target_db}";')
            else:
                print(f"[setup] database {target_db} already exists")


def render_migration(table: str, dim: int | None) -> str:
    """Read migrations/001_init.sql and rewrite chunk references to ``table``.

    Important: PostgreSQL only allows ONE bm25 index per relation, so the
    index names must also be produced with the table prefix. We rewrite
    ``chunks_bm25`` and ``chunks_embedding_hnsw`` first (full token
    match including the underscore), then any remaining standalone
    ``chunks`` word (the table name).
    """
    sql_path = Path(__file__).parent.parent / "src" / "db" / "migrations" / "001_init.sql"
    sql = sql_path.read_text(encoding="utf-8")
    # 1. Rewrite index names that include the table name as a prefix.
    sql = sql.replace("chunks_bm25", f"{table}_bm25")
    sql = sql.replace("chunks_embedding_hnsw", f"{table}_embedding_hnsw")
    # 2. Rewrite any remaining standalone ``chunks`` word to the new table.
    sql = re.sub(r"\bchunks\b", table, sql)
    if dim is not None:
        sql = sql.replace("embedding    vector,", f"embedding    vector({dim}),")
    else:
        sql = re.sub(
            r"CREATE INDEX IF NOT EXISTS [a-zA-Z0-9_]*_embedding_hnsw[^;]*;\n",
            f"-- HNSW index skipped for {table} (no dim available)\n",
            sql,
        )
    return sql


def probe_dim() -> int | None:
    """Probe Ollama to determine the configured embed model's native dim."""
    try:
        from src.embeddings.embedder import Embedder

        return len(Embedder().embed_chunk("dim probe"))
    except Exception as exc:
        print(f"[probe] could not reach Ollama at {settings.ollama_base_url}: {exc}")
        return None


def apply_migration(target_url: str, table: str, dim: int | None) -> None:
    import psycopg

    sql = render_migration(table, dim)
    with psycopg.connect(target_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


def main() -> int:
    args = parse_args()

    # 1. Make sure the target database exists.
    ensure_database_exists(args.db)

    # 2. Resolve the connection string for that database.
    target_url = db_url_for(args.db)
    print(f"[migrate] target = {target_url}")

    # 3. Probe embedding dim for the migration to substitute into vector().
    print("[probe] measuring embedding dim from Ollama ...")
    dim = probe_dim()
    print(f"[probe] dim = {dim}")

    # 4. Apply migration, but with chunks renamed to chunks_v2.
    apply_migration(target_url, args.table, dim)
    print(f"[migrate] table {args.table!r} ready in {args.db}")

    # 5. Call the existing indexer with our chosen chunker + table params.
    from src.ingestion.index import index_docs
    from pathlib import Path as P

    corpus = P(args.corpus)
    if not corpus.exists():
        print(f"[reindex] corpus not found: {corpus}")
        return 1

    chunker_kwargs = {
        "buffer_size": args.buffer_size,
        "breakpoint_percentile_threshold": args.threshold,
    }
    include_subdirs = tuple(args.include_subdir) if args.include_subdir else None
    print(
        f"[reindex] corpus={corpus} db={args.db} table={args.table} "
        f"include_subdirs={include_subdirs} chunker_kwargs={chunker_kwargs}"
    )

    total = index_docs(
        root_dir=str(corpus),
        db_url=target_url,
        table=args.table,
        chunker_kwargs=chunker_kwargs,
        include_subdirs=include_subdirs,
    )
    print(f"[reindex] done -- {total} chunks written into {args.db}.{args.table}")
    print(f"[reindex] tip: point your pipeline at {args.db} (e.g. .env DB_NAME=rag_v2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
