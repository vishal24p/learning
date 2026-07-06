-- ParadeDB BM25 + pgvector co-index for hybrid search.
-- Apply with: python scripts/apply_migration.py
--
-- pg_search >= 0.20 (newer API) requires:
--   - naming a key_field in WITH (key_field='<col>') on the BM25 index
--   - the column list is a comma-separated searchable field list
--
-- The embedding column needs an explicit dim for HNSW to be created. We
-- determine the dim at apply time by embedding a probe string with the
-- configured model and substitute it via `:dim`.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  TEXT        NOT NULL,
    chunk_index  INTEGER     NOT NULL,
    content      TEXT        NOT NULL,
    embedding    vector,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_bm25 ON chunks
USING bm25 (id, content)
WITH (key_field = 'id');

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks
USING hnsw (embedding vector_cosine_ops);

