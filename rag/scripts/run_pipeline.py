r"""One-shot end-to-end smoke test.

What this does, in order:
  1. Applies migrations/001_init.sql  (CREATE EXTENSION/TABLE/INDEX, idempotent)
  2. Indexes website-main/content/en/docs if the chunks table is empty
  3. Prints dense, sparse, and RRF-fused results for one query

Usage (Windows PowerShell):
    $env:PYTHONPATH = "C:\Users\visha\rag"
    python scripts\run_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from src.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from src.retrieval.sparse_retriever import SparseRetriever  # noqa: E402
from src.storage.postgres_store import PostgresStore  # noqa: E402

QUERY = "How do H1, H2, H3 heading levels work in markdown?"
CORPUS_DIR = "website-main/content/en/docs"


def apply_migration() -> None:
    sql = (Path(__file__).parent.parent / "src" / "db" / "migrations" / "001_init.sql").read_text(
        encoding="utf-8"
    )
    import psycopg

    with psycopg.connect(settings.db_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    print("[migrate] applied 001_init.sql")


def ensure_indexed() -> int:
    from src.ingestion.index import index_docs

    store = PostgresStore(settings.db_url)
    before = store.count()
    if before > 0:
        print(f"[index] chunks already present: {before} (skipping)")
        return before
    print(f"[index] indexing {CORPUS_DIR} ...")
    return index_docs(CORPUS_DIR, db_url=settings.db_url)


def show(name: str, hits: list[dict]) -> None:
    print(f"\n== {name} ({len(hits)} hits) ==")
    for i, h in enumerate(hits, 1):
        preview = h["content"].replace("\n", " ")[:140]
        print(f"  {i}. id={h['chunk_id']:<4}  score={h['score']:.5f}  | {preview}...")


def main() -> int:
    apply_migration()
    ensure_indexed()

    dense = DenseRetriever(settings.db_url)
    sparse = SparseRetriever(settings.db_url)
    hybrid = HybridRetriever(settings.db_url)

    show("DENSE ", dense.retrieve(QUERY, top_k=5))
    show("SPARSE", sparse.retrieve(QUERY, top_k=5))
    show("RRF   ", hybrid.retrieve(QUERY, top_k_dense=10, top_k_sparse=10, top_k=5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
