"""Single orchestrator: load -> chunk -> embed -> store (dense + sparse).

The hot loop used to read one doc at a time: chunk_text(...) -> embed_chunks(...)
-> store.insert_chunk(...) repeatedly. On the full K8s corpus that's about
2 hours of mostly waiting on Ollama.

We now do three things to bring that down:

1. Single chunker instance, reused (the parser caches intermediate state).
2. Embed baches in larger strides using ``Embedder._embed_texts``
   (one POST /api/embed containing the whole per-doc node list).
3. Insert chunks in batches of ``INSERT_BATCH_SIZE`` rows via
   ``PostgresStore.insert_chunks_many`` -- one psycopg round-trip per
   batch instead of one per row.

The Ollama client itself also has ``parallel=N`` enabled in
``build_embed_model`` so it can pipeline several concurrent batches.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time

from src.chunking.semantic_chunker import SemanticChunker
from src.config import settings
from src.embeddings.embedder import Embedder
from src.loaders.markdown_loader import load_markdown_files
from src.storage.postgres_store import PostgresStore


INSERT_BATCH_SIZE = int(os.getenv("RAG_INSERT_BATCH", "100"))
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH", "32"))


def _chunk_one(chunker: SemanticChunker, doc: dict) -> dict:
    nodes = chunker.chunk_text(doc["content"])
    return {"id": doc["id"], "path": doc["path"], "texts": [n.text for n in nodes]}


def index_docs(
    root_dir: str,
    db_url: str | None = None,
    *,
    table: str = "chunks",
    chunker_kwargs: dict | None = None,
    include_subdirs: tuple[str, ...] | None = None,
    exclude_dirs: tuple[str, ...] | None = None,
) -> int:
    """Index a tree of markdown files. Returns number of chunks written."""
    db_url = db_url or settings.db_url
    if exclude_dirs is None:
        docs = load_markdown_files(root_dir, include_subdirs=include_subdirs)
    else:
        docs = load_markdown_files(
            root_dir,
            include_subdirs=include_subdirs,
            exclude_dirs=exclude_dirs,
        )
    if not docs:
        print(f"No markdown files under {root_dir}")
        return 0

    kw = chunker_kwargs or {}
    chunker = SemanticChunker(**kw)
    embedder = Embedder()
    store = PostgresStore(db_url, table=table)

    print(
        f"[index] docs={len(docs)}  embed_batch={EMBED_BATCH_SIZE}  "
        f"insert_batch={INSERT_BATCH_SIZE}  table={table!r}"
    )

    total = 0
    flush: list[dict] = []

    def flush_insert() -> None:
        nonlocal flush
        if not flush:
            return
        store.insert_chunks_many(flush)
        flush = []

    for i, doc in enumerate(docs, 1):
        # 1. Chunk (single thread; SemanticSplitterNodeParser is stateful-ish).
        nodes = chunker.chunk_text(doc["content"])

        # 2. Embed the whole doc's nodes in batches of EMBED_BATCH_SIZE.
        embeddings: list[list[float]] = []
        text_list = [n.text for n in nodes]
        for j in range(0, len(text_list), EMBED_BATCH_SIZE):
            sub = text_list[j : j + EMBED_BATCH_SIZE]
            items = embedder.embed_chunks(sub)  # List[dict] of {text, embedding}
            embeddings.extend(item["embedding"] for item in items)

        # 3. Stage rows for batched insert.
        for k, (text, emb) in enumerate(zip(text_list, embeddings)):
            flush.append(
                {
                    "document_id": doc["id"],
                    "chunk_index": k,
                    "content": text,
                    "embedding": emb,
                }
            )
            if len(flush) >= INSERT_BATCH_SIZE:
                flush_insert()

        total += len(nodes)
        if i % 50 == 0:
            print(f"  ... {i}/{len(docs)} docs processed, chunks so far: {total}")
        if i % 200 == 0:
            print(f"    flush so far: {total - (len(flush))} rows committed")

    # 4. Drain any remaining rows.
    flush_insert()

    print(f"Indexed {total} chunks from {len(docs)} docs into table {table!r}")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG indexer")
    parser.add_argument("root", help="Directory of markdown files")
    args = parser.parse_args()
    return 0 if index_docs(args.root) >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
