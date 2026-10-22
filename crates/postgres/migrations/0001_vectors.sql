-- Durable storage for the pgvector-backed vector engine and the embedding
-- spend ceiling. Trimmed from the production migrations to the tables these
-- two adapters touch; the original also carried memory and graph tables and a
-- foreign key from `workspace_id` to a workspaces table.

CREATE EXTENSION IF NOT EXISTS vector;

-- ── generic vector engine ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS engine_vectors (
    workspace_id    UUID NOT NULL,
    namespace       TEXT NOT NULL CHECK (length(trim(namespace)) > 0),
    id              UUID NOT NULL,
    embedding_model TEXT,
    embedding       VECTOR(3072) NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (workspace_id, namespace, id)
);

CREATE INDEX IF NOT EXISTS engine_vectors_namespace_idx
    ON engine_vectors (workspace_id, namespace, updated_at DESC);
CREATE INDEX IF NOT EXISTS engine_vectors_metadata_idx
    ON engine_vectors USING GIN (metadata);
CREATE INDEX IF NOT EXISTS engine_vectors_embedding_idx
    ON engine_vectors USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops);

-- Vectors from more than one model, side by side.
--
-- Two embeddings of the same text by different models are not comparable. They
-- are the same width and mean unrelated things, so cosine similarity between
-- them is noise -- with no error, no dimension mismatch, and nothing in a log.
-- The symptom is a top result scoring 0.0161 and a search that looks like it
-- works.
--
-- A single embedding column forces that: changing model means every stored
-- vector is wrong until all of them have been redone, and the corpus is
-- unsearchable in between. This table holds one vector per (row, model)
-- instead, so a new model is backfilled alongside the old one and becomes
-- active only once it covers everything. Switching is then instant, and so is
-- switching back.
--
-- `embedding` is `vector` with no declared width on purpose. Models do not
-- agree on dimensionality -- 3072 for the large hosted models, 1024 for the
-- open-weight models likely to be self-hosted later -- and Postgres stores any
-- of them in an undeclared column. The width is pinned per model in the index
-- instead, which is where it actually has to be fixed.
CREATE TABLE IF NOT EXISTS engine_vector_embeddings (
    workspace_id UUID NOT NULL,
    namespace    TEXT NOT NULL,
    vector_id    UUID NOT NULL,
    model        TEXT NOT NULL,
    embedding    VECTOR NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, namespace, vector_id, model),
    FOREIGN KEY (workspace_id, namespace, vector_id)
        REFERENCES engine_vectors (workspace_id, namespace, id) ON DELETE CASCADE
);

-- One index per model, each pinning the width that model actually produces.
--
-- Partial on `model` so a 1024-wide model can be added later without disturbing
-- this one: its rows live in the same column and get an index of their own. The
-- cast to `halfvec` is the trick that makes 3072 dimensions indexable at all --
-- pgvector caps `vector` HNSW at 2000 dimensions and `halfvec` at 4000.
CREATE INDEX IF NOT EXISTS engine_vector_embeddings_openai_large_idx
    ON engine_vector_embeddings USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops)
    WHERE model = 'text-embedding-3-large';

-- ── embedding spend ────────────────────────────────────────────────────────

-- What has been spent on embeddings, so a budget can outlive a restart.
--
-- A row per provider and model rather than one total: the price differs
-- between them, and a cap counted in dollars has to know which rate each
-- token was charged at. Tokens are the provider's own reported figure, which
-- is what the bill is written from.
CREATE TABLE IF NOT EXISTS embedding_spend (
    provider    TEXT        NOT NULL,
    model       TEXT        NOT NULL,
    tokens      BIGINT      NOT NULL DEFAULT 0,
    requests    BIGINT      NOT NULL DEFAULT 0,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (provider, model)
);
